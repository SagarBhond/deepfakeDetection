import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.resnext_lstm import ResNextLSTMDeepfakeDetector, create_model
from data.preprocessing import create_data_loaders, get_transforms


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save the best model weights"""
        self.best_weights = model.state_dict().copy()


class DeepfakeTrainer:
    """Training class for deepfake detection model"""
    
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 device, 
                 learning_rate=1e-4,
                 weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, num_epochs=50, save_dir="models/checkpoints"):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(save_dir, 'final_model.pth'))
        
        # Save training history
        self.save_training_history(save_dir)
        
        # Generate plots
        self.plot_training_history(save_dir)
        
        # Generate classification report
        self.generate_classification_report(val_preds, val_targets, save_dir)
    
    def save_training_history(self, save_dir):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, save_dir):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_classification_report(self, predictions, targets, save_dir):
        """Generate and save classification report"""
        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Generate classification report
        report = classification_report(targets, predictions, 
                                     target_names=['Real', 'Fake'],
                                     output_dict=True)
        
        # Save report
        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=['Real', 'Fake']))


def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--real_paths', type=str, nargs='+', 
                       help='Paths to real video files')
    parser.add_argument('--fake_paths', type=str, nargs='+',
                       help='Paths to fake video files')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Number of frames in sequence')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='models/checkpoints',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if paths are provided
    if not args.real_paths or not args.fake_paths:
        print("Please provide paths to real and fake video files")
        print("Example: python train.py --real_paths data/real/*.mp4 --fake_paths data/fake/*.mp4")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        args.real_paths, args.fake_paths,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        sequence_length=args.sequence_length,
        num_classes=2,
        device=device
    )
    
    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
