from ultralytics import YOLO
import cv2
import numpy as np

class Model:

    def __init__(self):
        self.yolo_model = YOLO("yolo11l_half.engine")

        self.classes = [0] # humans
        self.tracked_id = None # remember to set before running self.track()
        self.last_position = None
        self.lost = False
        self.ignore_ids = []

        self.redetect_within = 50

    def show_all_boxes(self, image):
        """
        Get a jpg image of the frame with all detected bounding boxes with their ids
        To be used at the start to select the object to track

        params
        image (np.array) : BGR(A) array of image to analyze (From pyzed.sl.Mat().get_data())

        returns
        (bytes) jpg image with bounding boxes + ids
        """
        image = image[:, :, :3] # remove A channel from frame
        result = self.yolo_model.track(image, persist=True, classes=self.classes, verbose=False)[0]
        return self.np_to_jpeg(result.plot())


    def track(self, image):
        """
        Run the model on an image to keep track of and locate the tracked object
        If the tracked object disappeared from frame, will change mode to "lost" and will analyze new frames for any new object that:
        - Was not seen at the same time as the original object
        - Has a bounding box with similar location to the original object's last known position
        While lost, will return the input image with no bounding box
        
        When a matching object is detected, it will resume tracking that object and return the image with a bounding box

        params
        image (np.array) : BGR(A) array of image to analyze (From pyzed.sl.Mat().get_data())

        returns
        tensor([x1, y1, x2, y2]) : top left and bottom right coordinates of bounding box (of type float)
        """
        self.result = self.yolo_model.track(image[:, :, :3], persist=True, classes=self.classes, verbose=False)[0]
        if not self.lost:
            tracked_index = self._get_tracked_index()
            
            if tracked_index is False:
                # change mode to lost if tracked id not found
                self.lost = True
                return False
            else:
                # add other detected objects to list of ids to ignore (in case of tracked object lost)
                for i in self.result.boxes.id:
                    if i != self.tracked_id:
                        self.ignore_ids.append(i)
                
                self.last_position = self.result.boxes.xyxy[tracked_index]
                return self.last_position
        else:
            # if lost, wait for object with similar position to reappear
            if self.result.boxes.id is None:
                return False
            
            for i in range(len(self.result.boxes.id)):

                # ignore objects that appeared at the same time as original tracked object
                if self.result.boxes.id[i] not in self.ignore_ids:

                    coords = self.result.boxes.xyxy[i]
                    # print("new box id: " + str(self.result.boxes.id[i]))
                    # print("    coords: ", coords)
                    # print("last known: ", self.last_position)
                    corners = [
                        abs(coords[0] - self.last_position[0]) <= self.redetect_within,
                        abs(coords[1] - self.last_position[1]) <= self.redetect_within,
                        abs(coords[2] - self.last_position[2]) <= self.redetect_within,
                        abs(coords[3] - self.last_position[3]) <= self.redetect_within
                    ]
                    # print(corners)
                    # print(corners.count(True))
                    # print()
                    
                    # assume the object is the original if at least 3 corners are close to the last known position
                    if corners.count(True) >= 3:
                        self.lost = False
                        self.tracked_id = self.result.boxes.id[i]
                        return self.track(image)

            # if no suitable objects found
            return False

    def _get_tracked_index(self):
        if self.result.boxes.id is None:
            return False
        if self.tracked_id in self.result.boxes.id:
            return np.where(self.result.boxes.id.numpy() == self.tracked_id)[0][0]
        else:
            return False
    
    def np_to_jpeg(self, data):
        return bytes(cv2.imencode('.jpg', data)[1])






