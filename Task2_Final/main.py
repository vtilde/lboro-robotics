from ultralytics import YOLO
import cv2
import numpy as np


class Model:

    def __init__(self):
        """
        Initialise the tracking model and state.
        Loads the pretrained YOLO engine and sets up internal variables for tracking a single person
        across frames, including ID handling and redetection tolerance.
        """
        self.yolo_model = YOLO("yolo11n.engine")
        self.classes = [0] # Only track person class

        # Tracking state variables
        self.tracked_id = None
        self.last_position = None
        self.lost = False
        self.ignore_ids = []

        # Tolerance threshold for redetection if tracking is lost
        self.redetect_within = 50

    def show_all_boxes(self, image):
        """
        Run detection and return the image with all bounding boxes drawn.
        :param image (np.ndarray): The input image (BGR or BGRA).
        :return (bytes): JPEG-encoded image with boxes drawn for widget display.
        """
        image = image[:, :, :3] # remove A channel from frame
        result = self.yolo_model.track(image, persist=True, classes=self.classes, verbose=False)[0]
        return self.np_to_jpeg(result.plot())
    
    def start_tracking(self, image):
        """
        Begin tracking by selecting the most central detected person in the input image.
        :param image (np.ndarray): A frame from the camera, expected in BGR or BGRA format.
        :return (bool): True if a target is successfully selected and tracking is initialised,
                        False if no person is detected.
        """
        self.result = self.yolo_model.track(image[:, :, :3], persist=True, classes=self.classes, verbose=False)[0]
        if self.result.boxes.id is not None:
            # Pick the most central person by default
            centres = [(box[0] + box[2]) / 2 for box in self.result.boxes.xyxy]
            image_center_x = image.shape[1] / 2
            idx = np.argmin([abs(c - image_center_x) for c in centres])
            
            # Store that person's ID and bounding box
            self.tracked_id = self.result.boxes.id[idx].item()
            self.last_position = self.result.boxes.xyxy[idx]
            self.ignore_ids = [i for i in self.result.boxes.id if i != self.tracked_id]
            self.lost = False
            print(f"Started tracking ID {self.tracked_id}")
            return True
        return False
    
    def _compare_histogram(self, image, box1, box2, threshold=0.7):
        """
        Compare the colour similarity between two bounding boxes using histogram correlation.
        :param image (np.ndarray): The input image containing both bounding boxes.
        :param box1 (list or tuple): Coordinates (x1, y1, x2, y2) of the first bounding box.
        :param box2 (list or tuple): Coordinates (x1, y1, x2, y2) of the second bounding box.
        :param threshold (float): Similarity threshold for considering the boxes as a match. Default is 0.7.
        :return (bool): True if the histogram similarity exceeds the threshold, False otherwise.
        """
        # Extract histogram for a bounding box region in HSV colour space ===
        def get_hist(img, box):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        # Compute histograms
        hist1 = get_hist(image, box1)
        hist2 = get_hist(image, box2)

        # Compare histograms using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity > threshold


    def track(self, image, return_type="xcentre"):
        """
        Track the previously selected object in the current video frame.
        :param image (np.ndarray): The current frame captured from the camera (e.g., from sl.Mat().get_data()).
        :param return_type (str): Output format. Can be:
            - "centre": Returns (x_centre, y_centre) as a tuple
            - "corners": Returns (x1, y1, x2, y2) bounding box
            - "xcentre": Returns horizontal centre coordinate as an int
        :return (tuple or int or bool): Position data in the specified format,
            or False if tracking fails or redetection is unsuccessful.
        """
        self.result = self.yolo_model.track(image[:, :, :3], persist=True, classes=self.classes, verbose=False)[0]
        if not self.lost:
            tracked_index = self._get_tracked_index()
            
            if tracked_index is False:
                # Set as lost if tracked id not found
                self.lost = True
                return False
            else:
                # Add other detected objects to list of ids to ignore
                for i in self.result.boxes.id:
                    if i != self.tracked_id:
                        self.ignore_ids.append(i)
                
                self.last_position = self.result.boxes.xyxy[tracked_index]

                # check return type
                if return_type == "centre":
                    return (
                        int((self.last_position[0] + self.last_position[2]) / 2),
                        int((self.last_position[1] + self.last_position[3]) / 2)
                    )
                elif return_type == "corners":
                    return self.last_position
                elif return_type == "xcentre":
                    return int((self.last_position[0] + self.last_position[2]) / 2)
        else:
            # If lost, wait for object with similar position to reappear
            if self.result.boxes.id is None:
                return False
            
            for i in range(len(self.result.boxes.id)):

                # Ignore objects that appeared at the same time as original tracked object
                if self.result.boxes.id[i] not in self.ignore_ids:

                    coords = self.result.boxes.xyxy[i]
                    print("new box id: " + str(self.result.boxes.id[i]))
                    print("    coords: ", coords)
                    print("last known: ", self.last_position)
                    # TODO: not redetecting, make it more lenient
                    corners = [
                        abs(coords[0] - self.last_position[0]) <= self.redetect_within,
                        abs(coords[1] - self.last_position[1]) <= self.redetect_within,
                        abs(coords[2] - self.last_position[2]) <= self.redetect_within,
                        abs(coords[3] - self.last_position[3]) <= self.redetect_within
                    ]
                    print(corners)
                    print(corners.count(True))
                    print()
                    if corners.count(True) >= 3 and self._compare_histogram(image, coords, self.last_position):
                        self.lost = False
                        self.tracked_id = self.result.boxes.id[i]
                        return self.track(image, return_type=return_type)


            # if no suitable objects found
            return False

    def _get_tracked_index(self):
        """
        Get the index of the currently tracked object ID from the latest YOLO detection results.
        :return (int or bool): Index of the tracked object if found,
            or False if not found or no detections exist.
        """
        if self.result.boxes.id is None:
            return False
        if self.tracked_id in self.result.boxes.id:
            return np.where(self.result.boxes.id.numpy() == self.tracked_id)[0][0]
        else:
            return False
    
    def np_to_jpeg(self, data):
        """
        Convert a NumPy image array to JPEG-encoded bytes.
        :param data (np.ndarray): Image array in BGR format to encode.
        :return (bytes): JPEG-encoded image suitable for display in IPython widgets.
        """
        return bytes(cv2.imencode('.jpg', data)[1])