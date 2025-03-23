import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from datetime import datetime

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_dir='./models', callback=None):
    """
    Train the CNN model and return training history
    """
    # Create directory for saving model
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'time_per_epoch': []
    }
    
    best_val_loss = float('inf')
    
    # Train the model
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress every 10 batches
            if batch_idx % 10 == 0 and callback:
                train_acc = 100. * train_correct / train_total if train_total > 0 else 0
                progress_info = {
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'total_batches': len(train_loader),
                    'train_loss': train_loss / (batch_idx + 1),
                    'train_acc': train_acc
                }
                callback('progress', progress_info)
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save the model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, model_path)
            
            if callback:
                callback('save', {'path': model_path})
        
        # Calculate time per epoch
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s')
        
        # Callback with epoch results
        if callback:
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'time': epoch_time
            }
            callback('epoch', epoch_results)
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(save_dir, f"final_model_{timestamp}.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, final_model_path)
    
    if callback:
        callback('complete', {'path': final_model_path, 'history': history})
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    class_accuracy = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    
    return accuracy, class_accuracy