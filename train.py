# train.py

import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from models.hybrid_model import UQCCNN
from models.losses import FocalLoss
from utils.dataset import get_dataloaders
from utils.training import run_epoch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Task Setup -----
    if args.task == "binary":
        num_classes = 1
        criterion = FocalLoss()
    else:
        num_classes = 4
        criterion = nn.CrossEntropyLoss()

    model = UQCCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loader, val_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size
    )

    # ----- Training Loop -----
    for epoch in range(args.epochs):

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, args.task, train=True
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, args.task, train=False
        )

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    torch.save(model.state_dict(), f"uqccnn_{args.task}.pth")
    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    main()