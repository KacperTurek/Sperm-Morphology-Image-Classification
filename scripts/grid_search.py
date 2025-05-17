import itertools
import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms

param_grid = {
    'learning_rate': [1e-3, 7e-4, 5e-4],
    'dropout_fc_rate': [0.3, 0.35, 0.4]
}

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
    
def prepare_dataset(data_path, batch, model_path):
    classes = os.listdir(data_path)
    data = []
    for label in classes:
        folder_path = os.path.join(data_path, label)
        for file in os.listdir(folder_path):
            if file.endswith(('jpg','png','bmp')):  
                file_path = os.path.join(folder_path, file)
                data.append((file_path, label))
        
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    label_mapping = {'Normal_Sperm': 0, 'Abnormal_Sperm': 1, 'Non-Sperm': 2}
    df['label'] = df['label'].map(label_mapping)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify = df['label'])
    test_df, valid_df = train_test_split(temp_df, test_size=0.3, random_state=42, stratify = temp_df['label'])
    test_df.to_csv(f"{model_path}/test_dataset.csv", index=False)

    train_dataset = CustomDataset(train_df, transform=transform, c_type="L")
    valid_dataset = CustomDataset(valid_df, transform=transform, c_type="L")

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False)

    return train_loader, valid_loader
    

def train_with_params(run_id, num_epochs, patience, params, train_loader, valid_loader, model_path):
    run_name = f"grid_trial_{run_id}"
    print(f"\nTraining {run_name} with params: {params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(dropout_rate=params['dropout_fc_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        "Adam": torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0),
        "SGD": torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    }
    optimizer = optimizers["Adam"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_acc_history = []
    train_loss_history = []
    valid_acc_history = []
    valid_loss_history = []

    best_valid_acc = 0.0
    best_model_path = f"{model_path}/{run_name}.pt"
    early_stop_counter = 0

    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels = []
        all_preds = []

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            train_loss_avg = running_loss / (i + 1)
            train_accuracy = 100 * correct_train / total_train
            train_bar.set_postfix({'loss': f'{train_loss_avg:.3f}', 'acc': f'{train_accuracy:.2f}%'})

        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_precision = precision_score(all_labels, all_preds, average='macro')
        train_recall = recall_score(all_labels, all_preds, average='macro')
        train_loss_avg = running_loss / len(train_loader)
        train_loss_history.append(train_loss_avg)
        train_acc_history.append(train_accuracy)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss_avg:.3f}, Train Acc: {train_accuracy:.2f}%, '
            f'Train F1: {train_f1:.3f}, Train Precision: {train_precision:.3f}, Train Recall: {train_recall:.3f}')
        
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            valid_bar = tqdm(valid_loader, total=len(valid_loader), desc="Validation")
            for images, labels in valid_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                valid_loss_avg = valid_loss / len(valid_loader)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_bar.set_postfix({'loss': f'{valid_loss_avg:.3f}', 'acc': f'{valid_accuracy:.2f}%'})

        valid_f1 = f1_score(all_labels, all_preds, average='macro')
        valid_precision = precision_score(all_labels, all_preds, average='macro',zero_division=1)
        valid_recall = recall_score(all_labels, all_preds, average='macro',zero_division=1)
        valid_loss_avg = valid_loss / len(valid_loader)
        valid_loss_history.append(valid_loss_avg)
        valid_acc_history.append(valid_accuracy)

        print(f'Epoch {epoch + 1}, Valid Loss: {valid_loss_avg:.3f}, Valid Acc: {valid_accuracy:.2f}%, '
            f'Valid F1: {valid_f1:.3f}, Valid Precision: {valid_precision:.3f}, Valid Recall: {valid_recall:.3f}')
        print("Validation Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved: Valid Accuracy = {best_valid_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{patience}")

        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if early_stop_counter >= patience:
            print(f"Early stop triggered! No recovery during {patience} epoch.")
            break

    time_elapsed = time.time() - start
    print(f"Training finished after: {time_elapsed:.2f} seconds")

    return {**params, "run_name": run_name, "valid_acc": best_valid_acc}

def main(opt):
    data_path = opt.data
    epochs = opt.epochs
    patience = opt.patience
    batch = opt.batch
    model_path = opt.model_path
    print(f"Results will be saved to {model_path}/grid_search_results.csv")
    param_combos = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    train_loader, valid_loader = prepare_dataset(data_path, batch, model_path)

    results = []
    for i, combo in enumerate(param_combos):
        params = dict(zip(param_keys, combo))
        result = train_with_params(i, epochs, patience, params, train_loader, valid_loader, model_path)
        results.append(result)

    df = pd.DataFrame(results).sort_values(by="valid_acc", ascending=False)
    df.to_csv(f"{model_path}/grid_search_results.csv", index=False)

    print("Grid Search Complete. Top Results:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='models/grid_search_script')
    opt = parser.parse_args()

    main(opt)
