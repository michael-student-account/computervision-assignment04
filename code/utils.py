import torch
import torch.nn.functional as F


def semi_hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    Semi-hard triplet loss function.
    Adapted from a shell structure suggested by ChatGPT.
    """

    # Pre-calculate pairwise distances between embeddings to save compute
    distances = torch.cdist(embeddings, embeddings, p=2)

    batch_size = embeddings.size(0)
    losses = []

    # Iterate over all images in the batch, each serving as an anchor
    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]

        # Positive mask: same ID, but not the anchor itself
        positive_mask = labels == anchor_label
        positive_mask[anchor_idx] = False

        # Negative mask: different ID
        negative_mask = labels != anchor_label

        positive_indices = torch.where(positive_mask)[0]
        negative_indices = torch.where(negative_mask)[0]

        # Skip if no valid positive or negative
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue

        for positive_idx in positive_indices:
            d_ap = distances[anchor_idx, positive_idx]

            negative_distances = distances[anchor_idx, negative_indices]

            # Semi-hard condition:
            # d_ap < d_an < d_ap + margin
            semi_hard_mask = (
                (negative_distances > d_ap) &
                (negative_distances < d_ap + margin)
            )

            semi_hard_negatives = negative_indices[semi_hard_mask]

            if len(semi_hard_negatives) > 0:
                # Choose one semi-hard negative at random
                chosen_negative_idx = semi_hard_negatives[
                    torch.randint(
                        len(semi_hard_negatives),
                        size=(1,),
                        device=embeddings.device
                    )
                ]

                d_an = distances[anchor_idx, chosen_negative_idx]

                # Hinge loss for the triplet
                raw_loss = d_ap - d_an + margin
                triplet_loss = torch.clamp(raw_loss, min=0.0)
                losses.append(triplet_loss)

    if len(losses) == 0:
        # Return zero loss connected to graph
        return embeddings.sum() * 0.0

    return torch.stack(losses).mean()

# DEPRECATED TRAINING FUNCTION: WITH RANDOM TRIPLET SAMPLING AND TRIPLET LOSS
def train_embedding_model(model, dataloader, optimizer, device, loss_function):
    """
    This function trains the embedding model for one epoch.
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