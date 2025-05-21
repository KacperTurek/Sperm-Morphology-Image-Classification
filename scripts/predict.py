import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter()
])

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.4):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 112 -> 56
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 56 -> 28
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 28 -> 14
        x = x.view(-1, 64 * 14 * 14)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def prepare_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image

def predict(weights, model_path, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(dropout_rate=0.35).to(device)
    model.load_state_dict(torch.load(f"{model_path}/{weights}", weights_only=True))
    class_names = ['Normal_Sperm', 'Abnormal_Sperm', 'Non-Sperm'] 
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        print(f"Predicted class: {class_names[class_idx]}")
        return class_names[class_idx]

def main(opt):
    image_path = opt.image_path
    model_path = opt.model_path
    weights = opt.weights
    results_path = opt.results_path
    print(f"Results will be saved to {results_path}")
    image = prepare_image(image_path)
    results = predict(weights, model_path, image)
    df = pd.DataFrame([{"image_path": image_path, "class": results}])
    df.to_csv(f"{results_path}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='data/Normal_Sperm/Normal_Sperm (609).bmp', help='Path to image')
    parser.add_argument('--model_path', type=str, default='models/train_script')
    parser.add_argument('--weights', type=str, default='2025-05-19_12:15.pt')
    parser.add_argument('--results_path', type=str, default='results.csv')
    opt = parser.parse_args()

    main(opt)
