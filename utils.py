import os
import matplotlib.pyplot as plt

def plot_training_history(history, save_dir='results'):
    """Plot training and validation metrics.
    
    Args:
        history (dict): Dictionary containing training history
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a 1x2 grid of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot training & validation accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")
