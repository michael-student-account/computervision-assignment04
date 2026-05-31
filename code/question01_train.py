import torch
from question01_models import SimpleReIdModel
from data import create_dataloaders
from datetime import datetime
from pathlib import Path
import json
from tqdm.auto import tqdm
from utils import semi_hard_triplet_loss

def train_embedding_model(model, dataloader, optimizer, device, margin=0.3):
    """
    This function trains the embedding model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_images = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["vehicle_id"].to(device) 
        batch_size = images.shape[0]

        # Forward pass with all embeddings
        embeddings = model(images)

        # Triplet loss function
        loss = semi_hard_triplet_loss(embeddings, labels, margin=margin)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_images += batch_size

        running_avg_loss = total_loss / total_images

        progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{running_avg_loss:.4f}"
        })
    
    # Average loss over the epoch
    average_loss = total_loss / total_images

    return average_loss

def main():
    """
    Execute training and evaluation.
    """
    # HYPERPARAMETERS
    alpha = 0.2
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 50
    embedding_dim = 256
    P = 8
    K = 4
    image_resolution = 128

    # Initialize directory to save outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('training_outputs') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize paths
    model_save_path = output_dir / 'embedding_model_weights.pth'
    history_path = output_dir / 'training_history.json'

    training_history = {
        'triplet_loss': [],
        'hyperparameters': {
            'margin': alpha,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'P': P,
            'K': K,
            'embedding_dim': embedding_dim,
            'learning_rate': learning_rate,
            'image_resolution': image_resolution
        }
    }

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleReIdModel(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Dataloaders
    print("Loading data...")
    base_directory = Path(__file__).resolve().parent  # starting point is the code directory
    data_directory = base_directory.parent / 'data' / 'VeRi'
    train_loader, _, _ = create_dataloaders(data_directory, image_resolution, P=P, K=K)
    print("Data loaded successfully.")

    for epoch in range(num_epochs):
        # Start timer
        start_time = datetime.now()

        print(f"Epoch {epoch+1}/{num_epochs}")

        avg_loss = train_embedding_model(
            model, train_loader, optimizer, device, margin=alpha
        )
        training_history['triplet_loss'].append(avg_loss)

        # End timer and print epoch summary
        end_time = datetime.now()
        print(f"Epoch {epoch+1}/{num_epochs} - Triplet Loss: {avg_loss:.4f}."
              f"Time taken: {end_time - start_time}")
    
    # Save model weights
    torch.save(model.state_dict(), model_save_path)

    # Save training history
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)

if __name__ == "__main__":
    main()