# utils/training.py

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def run_epoch(model, loader, criterion, optimizer, device, task, train=True):

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(train):

        for images, labels in tqdm(loader):

            images = images.to(device)
            labels = labels.to(device)

            if task == "binary":
                labels = labels.float().unsqueeze(1)
            else:
                labels = labels.long()

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # Predictions
            if task == "binary":
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).int()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            else:
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    if task == "binary":
        f1 = f1_score(all_labels, all_preds)
    else:
        f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1