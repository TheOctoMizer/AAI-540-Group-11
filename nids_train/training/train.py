import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

def train_autoencoder(model, X_train, X_val, config, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=config.batch_size)

    history = []
    best_val = float("inf")
    start_time = time.time()

    print(f"Starting training for {config.epochs} epochs...")
    print(f"Batch size: {config.batch_size}, Learning rate: {config.lr}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print("-" * 80)

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        model.eval()
        val_errors = []
        val_loss_sum = 0.0
        val_num_batches = 0

        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon = model(x)
                mse = ((x - recon) ** 2).mean(dim=1)
                val_errors.extend(mse.cpu().numpy())
                val_loss_sum += mse.mean().item()
                val_num_batches += 1

        avg_train_loss = train_loss / num_batches
        avg_val_loss = sum(val_errors) / len(val_errors)
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_time": epoch_time,
            "total_time": total_time
        })

        best_val = min(best_val, avg_val_loss)

        # Log metrics for this epoch
        print(f"Epoch {epoch + 1:3d}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | "
              f"Total: {total_time:.1f}s | "
              f"Best Val: {best_val:.6f}")

        # Additional metrics every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            val_std = np.std(val_errors)
            val_median = np.median(val_errors)
            improvement = ((history[-2]["val_loss"] - avg_val_loss) / history[-2]["val_loss"] * 100) if epoch > 0 else 0
            
            print(f"         Val Stats - Median: {val_median:.6f}, Std: {val_std:.6f}, Improvement: {improvement:+.2f}%")

    print("-" * 80)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best validation loss: {best_val:.6f}")

    return model, history, val_errors