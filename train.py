import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.preprocessor import HSIPreprocessor
from src.models import HSSNB
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import argparse
import os
from typing import Dict
import time
import random

class HSIDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            data: Tensor of shape (num_samples, 1, spectral_bands, height, width)
            labels: Tensor of shape (num_samples,)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """
    Compute OA, AA, and Kappa metrics
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
    Returns:
        Dictionary with 'OA', 'AA', 'Kappa' keys
    """
    oa = np.sum(y_true == y_pred) / len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {'OA': oa, 'AA': aa, 'Kappa': kappa, 'confusion_matrix': cm}

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)  # shape: (batch_size, 1, spectral_bands, height, width)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, criterion, val_loader, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    return epoch_loss, metrics

def main():
    parser = argparse.ArgumentParser(description='Train HSSNB model on Hyperspectral Data')
    parser.add_argument('--dataset', type=str, required=True, choices=['IP', 'PU', 'SA'], help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset .mat file')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth .mat file')
    parser.add_argument('--n_components', type=int, default=30, help='Number of PCA components')
    parser.add_argument('--window_size', type=int, default=25, help='Spatial window size')
    parser.add_argument('--remove_water_bands', action='store_true', help='Remove water absorption bands')
    parser.add_argument('--train_size', type=float, default=0.3, help='Training data proportion')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare data
    preprocessor = HSIPreprocessor(
        data_path=args.data_path,
        gt_path=args.gt_path,
        dataset_name=args.dataset,
        n_components=args.n_components,
        window_size=args.window_size,
        remove_water_bands=args.remove_water_bands
    )

    split_data = preprocessor.preprocess_pipeline(train_size=args.train_size)

    num_classes = int(torch.max(split_data['y_train'])) + 1

    # Create Datasets and DataLoaders
    train_dataset = HSIDataset(split_data['X_train'], split_data['y_train'].long())
    test_dataset = HSIDataset(split_data['X_test'], split_data['y_test'].long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = HSSNB(num_classes=num_classes)
    model = model.to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # For saving the best model
    best_oa = 0.0

    # Lists to track metrics
    train_losses = []
    val_losses = []
    oa_list = []
    aa_list = []
    kappa_list = []

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss = train(model, criterion, optimizer, train_loader, device)
        val_loss, metrics = validate(model, criterion, test_loader, device, num_classes)
        elapsed_time = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        oa_list.append(metrics['OA'])
        aa_list.append(metrics['AA'])
        kappa_list.append(metrics['Kappa'])

        print(f"Epoch [{epoch}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"OA: {metrics['OA']*100:.2f}% "
              f"AA: {metrics['AA']*100:.2f}% "
              f"Kappa: {metrics['Kappa']:.4f} "
              f"Time: {elapsed_time:.2f}s")

        # Save model if OA improves
        if metrics['OA'] > best_oa:
            best_oa = metrics['OA']
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            checkpoint_path = os.path.join(args.save_dir, f"best_model_{args.dataset}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Training complete.")

    # Save training metrics
    metrics_path = os.path.join(args.save_dir, f"metrics_{args.dataset}.npz")
    np.savez(metrics_path, train_losses=train_losses, val_losses=val_losses,
             oa_list=oa_list, aa_list=aa_list, kappa_list=kappa_list)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
