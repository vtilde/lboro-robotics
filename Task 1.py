import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image


class PathDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load labels
        self.labels = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                self.labels.append((parts[0], float(parts[1])))  # (filename, steering angle)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, steering_angle = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([steering_angle], dtype=torch.float32)



# Resize to 224 x 224 for ResNet15
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#train_dataset = PathDataset(image_dir="data/train_images", label_file="data/train_labels.txt", transform=transform)
#test_dataset = PathDataset(image_dir="data/test_images", label_file="data/test_labels.txt", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ResNet15 model
class PathFollowingCNN(nn.Module):
    def __init__(self):
        super(PathFollowingCNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: Steering angle
        )

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PathFollowingCNN().to(device)
criterion = nn.MSELoss()  # Regression loss for steering angles
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10  # You can increase this for better performance

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, angles in train_loader:
        images, angles = images.to(device), angles.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")