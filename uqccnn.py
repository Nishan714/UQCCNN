!pip install torch torchvision torchaudio pennylane matplotlib seaborn numpy tqdm scikit-learn --quiet

import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
import gc
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Configuration
DATASET_PATH = input("\n👉 Dataset path: ").strip()
USE_MIXED_PRECISION = True
USE_TTA = True
USE_MIXUP = True
BATCH_SIZE = 32

out_dir = "/content/HQCNN_Results"
os.makedirs(out_dir, exist_ok=True)
drive_out = None
if os.path.exists('/content/drive'):
    try:
        drive_out = os.path.join('/content/drive/MyDrive', 'HQCNN_Results')
        os.makedirs(drive_out, exist_ok=True)
    except:
        pass

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
else:
    USE_MIXED_PRECISION = False

print(f"✅ Device: {device}")

# Dataset Verification
def verify_dataset(base_path, out_dir_local):
    if not os.path.exists(base_path):
        return False, None, None
    
    train_dir = test_dir = None
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            if folder.lower() in ['training', 'train']:
                train_dir = folder_path
            elif folder.lower() in ['testing', 'test']:
                test_dir = folder_path
    
    if not train_dir or not test_dir:
        return False, None, None
    
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    test_classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    train_counts, test_counts = [], []
    for cls in train_classes:
        train_counts.append(len([f for f in os.listdir(os.path.join(train_dir, cls))
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
    for cls in test_classes:
        test_counts.append(len([f for f in os.listdir(os.path.join(test_dir, cls))
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
    
    # Save distribution
    df = pd.DataFrame({
        'Class': train_classes,
        'Train': train_counts,
        'Test': test_counts
    })
    df.to_csv(os.path.join(out_dir_local, "class_distribution.csv"), index=False)
    
    return True, train_dir, test_dir

result = verify_dataset(DATASET_PATH, out_dir)
if not result[0]:
    raise SystemExit("❌ Dataset verification failed!")
train_dir, test_dir = result[1], result[2]

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_data = datasets.ImageFolder(train_dir, transform=train_transform)
full_test_data = datasets.ImageFolder(test_dir, transform=test_transform)
all_classes = full_train_data.classes
n_classes = len(all_classes)

# Quantum Circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (3, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        return self.qlayer(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class HybridQCCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.is_binary = (num_classes == 1)
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attn1 = ChannelAttention(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = ChannelAttention(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attn3 = ChannelAttention(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attn4 = ChannelAttention(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.attn5 = ChannelAttention(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.35)
        
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc_quantum = nn.Linear(256, n_qubits)
        self.quantum = QuantumLayer()
        
        self.fc_fusion = nn.Linear(256 + n_qubits, 128)
        self.bn_fusion = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.attn1(self.bn1(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.attn2(self.bn2(self.conv2(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.attn3(self.bn3(self.conv3(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.attn4(self.bn4(self.conv4(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.attn5(self.bn5(self.conv5(x)))))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        classical_features = F.relu(self.bn_fc1(self.fc1(x)))
        classical_features = self.dropout(classical_features)
        
        quantum_input = torch.tanh(self.fc_quantum(classical_features))
        quantum_features = self.quantum(quantum_input).float()
        
        fused = torch.cat([classical_features, quantum_features], dim=1)
        fused = F.relu(self.bn_fusion(self.fc_fusion(fused)))
        fused = self.dropout(fused)
        
        return self.fc_out(fused)

# Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training Function
def train_model(model, model_name, train_loader, test_loader, task_type='binary',
                class_weights=None, num_epochs=80, patience=15):
    
    criterion = FocalLoss() if task_type == 'binary' else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    scaler = GradScaler() if (USE_MIXED_PRECISION and device == 'cuda') else None
    
    best_val_acc = 0
    epochs_no_improve = 0
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []
    learning_rates = []
    
    for epoch in range(num_epochs):
        if epochs_no_improve >= patience:
            break
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1) if task_type == 'binary' else labels.to(device)
            
            if USE_MIXUP and np.random.rand() < 0.5:
                if task_type == 'multiclass':
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels.unsqueeze(1).float())
                    labels_a = labels_a.squeeze(1).long()
                    labels_b = labels_b.squeeze(1).long()
                else:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                
                optimizer.zero_grad()
                if USE_MIXED_PRECISION and device == "cuda":
                    with autocast():
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if task_type == 'binary':
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (lam * (predicted == labels_a).sum().item() +
                               (1-lam) * (predicted == labels_b).sum().item())
                else:
                    _, predicted = torch.max(outputs, 1)
                    correct += (lam * (predicted == labels_a).sum().item() +
                               (1-lam) * (predicted == labels_b).sum().item())
            else:
                optimizer.zero_grad()
                if USE_MIXED_PRECISION and device == "cuda":
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if task_type == 'binary':
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (predicted == labels).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
            
            scheduler.step()
            running_loss += loss.item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = (correct / total) * 100
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) if task_type == 'binary' else labels.to(device)
                
                if USE_MIXED_PRECISION and device == "cuda":
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                if task_type == 'binary':
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                
                val_total += labels.size(0)
        
        val_loss /= len(test_loader)
        val_acc = (val_correct / val_total) * 100
        
        if val_acc > best_val_acc + 0.1:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_{model_name}_{task_type}.pth"))
        else:
            epochs_no_improve += 1
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train {train_acc:.2f}%, Val {val_acc:.2f}%")
        
        if device == "cuda" and epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    model.load_state_dict(torch.load(os.path.join(out_dir, f"best_{model_name}_{task_type}.pth")))
    return model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, learning_rates, best_val_acc

# Evaluation Function
def evaluate_model(model, test_loader, device, model_name, task_type='binary', class_names=None):
    model.eval()
    y_pred_list, y_true_list, y_pred_probs_list = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if USE_TTA:
                outputs = model(inputs)
                outputs_hflip = model(torch.flip(inputs, dims=[3]))
                outputs_vflip = model(torch.flip(inputs, dims=[2]))
                
                if task_type == 'binary':
                    probs = torch.sigmoid(outputs)
                    probs_hflip = torch.sigmoid(outputs_hflip)
                    probs_vflip = torch.sigmoid(outputs_vflip)
                    avg_probs = (probs + probs_hflip + probs_vflip) / 3
                    predicted = (avg_probs > 0.5).float()
                    y_pred_probs_list.extend(avg_probs.cpu().numpy().flatten())
                else:
                    probs = F.softmax(outputs, dim=1)
                    probs_hflip = F.softmax(outputs_hflip, dim=1)
                    probs_vflip = F.softmax(outputs_vflip, dim=1)
                    avg_probs = (probs + probs_hflip + probs_vflip) / 3
                    _, predicted = torch.max(avg_probs, 1)
                    y_pred_probs_list.extend(avg_probs.cpu().numpy())
            else:
                outputs = model(inputs)
                if task_type == 'binary':
                    avg_probs = torch.sigmoid(outputs)
                    predicted = (avg_probs > 0.5).float()
                    y_pred_probs_list.extend(avg_probs.cpu().numpy().flatten())
                else:
                    avg_probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(avg_probs, 1)
                    y_pred_probs_list.extend(avg_probs.cpu().numpy())
            
            y_pred_list.extend(predicted.cpu().numpy().flatten() if task_type == 'binary' else predicted.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
    
    return np.array(y_true_list), np.array(y_pred_list), np.array(y_pred_probs_list)

# Binary Dataset
class BinaryDataset(Dataset):
    def __init__(self, original_dataset, tumor_classes):
        self.original_dataset = original_dataset
        self.binary_labels = []
        for idx in range(len(original_dataset)):
            _, label = original_dataset[idx]
            self.binary_labels.append(1 if label in tumor_classes else 0)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.original_dataset[idx]
        return img, self.binary_labels[idx]

# Binary Classification Setup
tumor_class_indices = [i for i, cls in enumerate(all_classes) if cls.lower() != 'notumor']
train_binary = BinaryDataset(full_train_data, tumor_class_indices)
test_binary = BinaryDataset(full_test_data, tumor_class_indices)

train_loader_binary = DataLoader(train_binary, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader_binary = DataLoader(test_binary, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
classes_binary = ['no tumor', 'tumor']

# Train Binary Model
print("\n🚀 Training Binary Classification...")
model_binary = HybridQCCNN(num_classes=1).to(device)
results_binary = train_model(model_binary, "HQCNN", train_loader_binary, test_loader_binary,
                             task_type='binary', num_epochs=80, patience=15)
model_binary, train_acc_binary, val_acc_binary, train_loss_binary, val_loss_binary, lr_binary, best_acc_binary = results_binary

# Evaluate Binary Model
y_true_binary, y_pred_binary, y_probs_binary = evaluate_model(
    model_binary, test_loader_binary, device, "HQCNN", 'binary', classes_binary
)

print(f"Binary Accuracy: {accuracy_score(y_true_binary, y_pred_binary)*100:.2f}%")

# Multi-Class Setup
train_loader_multi = DataLoader(full_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader_multi = DataLoader(full_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Train Multi-Class Model
print("\n🚀 Training Multi-Class Classification...")
model_multi = HybridQCCNN(num_classes=n_classes).to(device)
results_multi = train_model(model_multi, "HQCNN", train_loader_multi, test_loader_multi,
                            task_type='multiclass', num_epochs=80, patience=15)
model_multi, train_acc_multi, val_acc_multi, train_loss_multi, val_loss_multi, lr_multi, best_acc_multi = results_multi

# Evaluate Multi-Class Model
y_true_multi, y_pred_multi, y_probs_multi = evaluate_model(
    model_multi, test_loader_multi, device, "HQCNN", 'multiclass', all_classes
)

print(f"Multi-Class Accuracy: {accuracy_score(y_true_multi, y_pred_multi)*100:.2f}%")

# Final Summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Binary Accuracy:      {accuracy_score(y_true_binary, y_pred_binary)*100:.2f}%")
print(f"Multi-Class Accuracy: {accuracy_score(y_true_multi, y_pred_multi)*100:.2f}%")
print("="*70)

if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()

print("\n✅ Training Complete!")