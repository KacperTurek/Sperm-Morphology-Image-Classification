import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter()
])

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, c_type="RGB"):
        self.df = df
        self.transform = transform
        self.c_type = c_type
        self.classes = df["label"].unique

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, id):
        img_path = self.df.iloc[id]["file_path"]
        label = self.df.iloc[id]["label"]
        if self.c_type.upper() == "RGB":
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert(self.c_type)
            
        if self.transform:
            image = self.transform(image)

        return image, label

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
    
def prepare_dataset(dataset_path, batch):
    test_df = pd.read_csv(dataset_path)

    test_dataset = CustomDataset(test_df, transform=transform, c_type="L")
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    return test_loader

def evaluate_model(model_name, model_path, data_loader, dropout_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(f"{model_path}/{model_name}", weights_only=True))
    criterion = nn.CrossEntropyLoss()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
            all_test_labels.extend(labels.detach().cpu().numpy())
            all_test_preds.extend(preds.detach().cpu().numpy())
            all_test_probs.extend(probs.detach().cpu().numpy())

    test_loss = test_loss / len(data_loader)
    test_accuracy = 100 * correct_test / total_test
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')

    test_auc = roc_auc_score(
        np.eye(3)[all_test_labels],   
        np.array(all_test_probs),     
        multi_class='ovr'
    )

    print("\nTest Results:")
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test F1 Score: {test_f1:.3f}')
    print(f'Test ROC AUC: {test_auc:.3f}')
    print("Test Confusion Matrix:\n", confusion_matrix(all_test_labels, all_test_preds))  

    cm = confusion_matrix(all_test_labels, all_test_preds)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=['Normal_Sperm', 'Abnormal_Sperm','Non-Sperm'], yticklabels=['Normal_Sperm', 'Abnormal_Sperm','Non-Sperm'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_path}/{model_name}_normalized_confusion_matrix.png')
    
    class_names = ['Normal_Sperm', 'Abnormal_Sperm', 'Non-Sperm']
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)

    plt.figure(figsize=(8, 6))

    for i in range(3): 
        binary_labels = (all_test_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f'{model_path}/{model_name}_roc_curve.png')

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "f1": test_f1,
        "auc": test_auc
    }

def main(opt):
    data_path = opt.dataset
    batch = opt.batch
    dropout_rate = opt.dropout_rate
    model_path = opt.model_path
    model_name = opt.model_name
    print(f"Results will be saved to {model_path}/{model_name}_evaluation_results.csv")
    test_loader = prepare_dataset(data_path, batch)
    results = evaluate_model(model_name, model_path, test_loader, dropout_rate)
    df = pd.DataFrame([results])
    df.to_csv(f"{model_path}/{model_name}_evaluation_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='models/train_script/2025-05-19_12:15_test_dataset.csv', help='Path to dataset')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--dropout_rate', type=int, default=0.35)
    parser.add_argument('--model_path', type=str, default='models/train_script')
    parser.add_argument('--model_name', type=str, default='2025-05-19_12:15.pt')
    opt = parser.parse_args()

    main(opt)
