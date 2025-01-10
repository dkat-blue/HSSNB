# <evaluate.py>
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.models import HSSNB
from src.preprocessor import HSIPreprocessor

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                          cmap='Blues', save_path=None):
    """
    Plots and optionally saves the confusion matrix using seaborn heatmap.
    Args:
        cm (np.ndarray): Confusion matrix of shape (num_classes, num_classes)
        class_names (list): Class labels for the axes
        title (str): Title for the heatmap
        cmap (str): Colormap to use
        save_path (str): Path to save the plot image. If None, the plot is not saved.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_curves(metrics_npz_path, save_dir):
    """
    Loads and plots training/validation losses and OA/AA/Kappa curves 
    from the metrics .npz file, then saves them as images.
    Args:
        metrics_npz_path (str): Path to the metrics_<dataset>.npz file
        save_dir (str): Directory where images will be saved
    """
    metrics_data = np.load(metrics_npz_path)
    
    train_losses = metrics_data['train_losses']
    val_losses = metrics_data['val_losses']
    oa_list = metrics_data['oa_list']
    aa_list = metrics_data['aa_list']
    kappa_list = metrics_data['kappa_list']
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    plt.close()
    
    # Plot OA, AA, Kappa
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, oa_list, label='OA')
    plt.plot(epochs, aa_list, label='AA')
    plt.plot(epochs, kappa_list, label='Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('OA, AA & Kappa Over Epochs')
    plt.legend()
    plt.tight_layout()
    metrics_plot_path = os.path.join(save_dir, 'metrics_curve.png')
    plt.savefig(metrics_plot_path)
    print(f"OA, AA & Kappa plot saved to {metrics_plot_path}")
    plt.close()

def load_best_model(checkpoint_path, num_classes):
    """
    Loads the saved model weights from checkpoint_path.
    Args:
        checkpoint_path (str): Path to the .pth file containing model weights
        num_classes (int): Number of classes
    Returns:
        model (torch.nn.Module): HSSNB model loaded with checkpoint weights
    """
    model = HSSNB(num_classes=num_classes)
    # Future PyTorch versions may require weights_only=True if you only need weights.
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def evaluate_on_full_testset(model, test_data_dict, device='cpu'):
    """
    Runs inference on the entire test set (X_test, y_test).
    Returns the arrays of ground-truth labels and predicted labels.
    Args:
        model (nn.Module): Trained model
        test_data_dict (dict): Contains 'X_test' and 'y_test' Tensors
        device (str): 'cpu' or 'cuda'
    Returns:
        (np.ndarray, np.ndarray):
            y_true: Ground truth labels (shape: [num_test_samples])
            y_pred: Predicted labels (same shape)
    """
    X_test = test_data_dict['X_test'].to(device)
    y_test = test_data_dict['y_test'].to(device)
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
    
    return y_test.cpu().numpy(), preds.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Evaluate and Visualize HSSNB Results (Option 2)')
    parser.add_argument('--dataset', type=str, required=True, choices=['IP', 'PU', 'SA'],
                        help='Dataset name for evaluation.')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory where model checkpoints and metrics are saved.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation ("cpu" or "cuda").')
    
    # Preprocessor arguments
    parser.add_argument('--data_path', type=str,
                        default='data/ip/Indian_pines_corrected.mat',
                        help='Path to the HSI .mat file for the dataset.')
    parser.add_argument('--gt_path', type=str,
                        default='data/ip/Indian_pines_gt.mat',
                        help='Path to the ground truth .mat file.')
    parser.add_argument('--n_components', type=int, default=30,
                        help='Number of PCA components to keep.')
    parser.add_argument('--window_size', type=int, default=25,
                        help='Spatial window size for patches.')
    parser.add_argument('--train_size', type=float, default=0.3,
                        help='Training data proportion used during the split.')
    
    # IMPORTANT: Default is now False so we do NOT remove bands again
    parser.add_argument('--remove_water_bands', type=bool, default=False,
                        help='Whether to remove water absorption bands. '
                             'Set to False if your data is already water-band-free.')
    
    args = parser.parse_args()
    
    # Paths to checkpoints and metrics
    checkpoint_path = os.path.join(args.checkpoints_dir, f"best_model_{args.dataset}.pth")
    metrics_path = os.path.join(args.checkpoints_dir, f"metrics_{args.dataset}.npz")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # Make a directory to store evaluation results and plots
    eval_results_dir = os.path.join(args.checkpoints_dir, 'eval_results', args.dataset)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 1. Plot training/validation losses and OA/AA/Kappa from the .npz file
    print(f"Loading metrics from: {metrics_path}")
    plot_training_curves(metrics_path, eval_results_dir)
    
    # 2. Load the best model
    #    We need the number of classes. For Indian Pines, commonly 16 classes.
    #    Adjust as needed for other datasets.
    num_classes_guess = 16 if args.dataset == 'IP' else 9 if args.dataset == 'PU' else 16
    print(f"Loading best model from: {checkpoint_path} with {num_classes_guess} classes.")
    model = load_best_model(checkpoint_path, num_classes=num_classes_guess)
    
    # 3. Reconstruct the test dataset with the same preprocessor settings
    print("Reconstructing the test set with the same preprocessing pipeline...")
    preprocessor = HSIPreprocessor(
        data_path=args.data_path,
        gt_path=args.gt_path,
        dataset_name=args.dataset,
        n_components=args.n_components,
        window_size=args.window_size,
        remove_water_bands=args.remove_water_bands  # <-- Now defaults to False
    )
    split_data = preprocessor.preprocess_pipeline(train_size=args.train_size)
    
    test_data_dict = {
        'X_test': split_data['X_test'],
        'y_test': split_data['y_test']
    }
    
    # 4. Evaluate model on the test set
    print("Evaluating the best model on the test set...")
    y_true, y_pred = evaluate_on_full_testset(model, test_data_dict, device=args.device)
    
    # 5. Compute and plot a fresh confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes_guess))
    print("\nFresh Confusion Matrix from re-inference:")
    print(cm)
    
    class_names = [f"Class {i+1}" for i in range(num_classes_guess)]
    cm_plot_path = os.path.join(eval_results_dir, 'fresh_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names,
                          title=f'Fresh Confusion Matrix - {args.dataset}',
                          save_path=cm_plot_path)
    
    print("\nEvaluation & visualization complete.")
    print(f"All plots saved to: {eval_results_dir}")


if __name__ == '__main__':
    main()
