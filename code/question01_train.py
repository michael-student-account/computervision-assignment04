import torch
from question01_models import SimpleReIdModel
from data import create_dataloaders
from datetime import datetime
from pathlib import Path
import json
from tqdm.auto import tqdm

def train_embedding_model(model, dataloader, optimizer, loss_function, device):
    """
    This function trains the embedding model for one epoch using triplet loss.
    """
    model.train()

    total_loss = 0.0
    total_triplets = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for batch in progress_bar:
        batch_size = batch['image'].shape[0]

        # Extract anchor, positive, and negative images from the dataloader
        anchor_image = batch['image'].to(device)  # [batch_size, 3, 256, 256]
        positive_image = batch['pos_anchor'].to(device)
        negative_image = batch['neg_anchor'].to(device)

        # Concatenate anchor, positive, and negative images along the batch dimension
        # [3*batch_size, 3, 256, 256]
        all_images = torch.cat(
            [anchor_image, positive_image, negative_image], 
            dim=0
        )

        # Forward pass with all embeddings
        all_embeddings = model(all_images)  # [3*batch_size, embedding_dim]

        # Split the embeddings back into anchor, positive, and negative
        anchor_embedding, positive_embedding, negative_embedding = torch.chunk(
            all_embeddings, 
            chunks=3,
            dim=0
        )

        # Triplet loss function
        loss = loss_function(
            anchor_embedding, positive_embedding, negative_embedding
        )

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_triplets += batch_size

        running_avg_loss = total_loss / total_triplets

        progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{running_avg_loss:.4f}"
        })
    
    # Average loss over the epoch
    average_loss = total_loss / total_triplets

    return average_loss

def main():
    """
    Execute training and evaluation.
    """
    # HYPERPARAMETERS
    alpha = 0.3
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 100
    embedding_dim = 256
    batch_size = 8
    image_resolution = 64

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
            'weight_dacay': weight_decay,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
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

    # Loss function: p = use L2 euclidean distance
    loss_function = torch.nn.TripletMarginLoss(margin=alpha, p=2)

    # Dataloaders
    print("Loading data...")
    base_directory = Path(__file__).resolve().parent  # starting point is the code directory
    data_directory = base_directory.parent / 'data' / 'VeRi'
    train_loader, _, _ = create_dataloaders(
        data_directory, image_resolution, batch_size=batch_size
    )
    print("Data loaded successfully.")

    for epoch in range(num_epochs):
        # Start timer
        start_time = datetime.now()

        print(f"Epoch {epoch+1}/{num_epochs}")

        avg_loss = train_embedding_model(
            model, train_loader, optimizer, loss_function, device
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