"""
Trainer class for model training.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

class ModelTrainer:
    """Trainer for Audio Classification"""
    
    def __init__(
            self,
            model: nn.Module,
            config: dict,
            device: torch.device,
            output_dir: Path
        ):
        """
        Args:
            model: PyTorch Model
            config: Training Configuration
            device: CPU or GPU
            output_dir: Output directory
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate warmup
        self.warmup_epochs = config['training'].get('warmup_epochs', 5)
        self.initial_lr = config['training']['learning_rate']
        self.current_epoch = 0
        
        # Scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            return 1.0
            
        self.warmup_scheduler = LambdaLR(
            self.optimizer, lr_lambda
        )
        
        # Scheduler with reduced factor and increased patience
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,  # Reduced from 0.5
            patience=8,   # Increased from 5
            verbose='True'  # Changed to boolean type
        )
        
        # Loss with class weights if specified
        if 'class_weights' in config['training']:
            weights = torch.tensor(config['training']['class_weights']).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Using class weights: {weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Best model metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Gradient clipping value
        self.grad_clip = config.get('gradient_clip', None)
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(
            self,
            train_loader: DataLoader
        ) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress Bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Move data to device
            inputs = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update Progress Bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        # Epoch Statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
            self,
            val_loader: DataLoader
        ) -> Tuple[float, float]:
        """Validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = [0] * self.config['num_classes']
        class_total = [0] * self.config['num_classes']
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                inputs = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward Pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Validation Statistics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Log per-class accuracy
        for i in range(self.config['num_classes']):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                self.logger.info(f'Accuracy of class {i}: {class_acc:.2f}%')
        
        return avg_loss, accuracy
    
    def save_checkpoint(
            self,
            epoch: int,
            val_loss: float,
            is_best: bool = False
        ) -> None:
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Regular checkpoint
        torch.save(
            checkpoint,
            self.output_dir / 'last_checkpoint.pt'
        )
        
        # Best model
        if is_best:
            torch.save(
                checkpoint,
                self.output_dir / 'best_model.pt'
            )
    
    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: Optional[int] = None
        ) -> Dict[str, list]:
        """Perform training"""
        # Ensure num_epochs is an integer
        epochs: int = self.config['num_epochs'] if num_epochs is None else num_epochs
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Learning rate warmup
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(history['val_loss'][-1])
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Logging
            self.logger.info(
                f"Training Loss: {train_loss:.4f}, "
                f"Acc: {train_acc:.2f}%"
            )
            self.logger.info(
                f"Validation Loss: {val_loss:.4f}, "
                f"Acc: {val_acc:.2f}%"
            )
            
            # Save model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early Stopping
            if self.patience_counter >= self.config['early_stopping']['patience']:
                self.logger.info(
                    f"Early Stopping after {epoch + 1} epochs"
                )
                break
        
        return history
