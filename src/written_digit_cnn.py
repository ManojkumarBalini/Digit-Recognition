import click
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from cnn_model import SmallCNN


# YOUR NAME: [Your Name]
# YOUR WPI ID: [Your WPI ID]


# ---------------------------
# Dataset
# ---------------------------
class DigitImageDataset(Dataset):
    """
    Expects images.npy with shape (N, 1, 28, 28) and labels.npy with shape (N,)
    Returns (image_tensor, label) where image_tensor is shape (1, 28, 28) in [0,1].
    """
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)  # (N, 1, 28, 28)
        self.labels = np.load(labels_path).astype(np.int64)  # (N,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)  # (1,28,28)
        label = self.labels[idx]

        # scale to [0,1] if needed
        if img.max() > 1.0:
            img = img / 255.0

        img_tensor = torch.from_numpy(img)        # (1,28,28)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor


# ---------------------------
# Data prep
# ---------------------------
def prepare_dataloaders(data_dir, batch_size, fabric):
    train_images = data_dir / "train_images.npy"
    train_labels = data_dir / "train_labels.npy"
    val_images = data_dir / "val_images.npy"
    val_labels = data_dir / "val_labels.npy"

    train_ds = DigitImageDataset(train_images, train_labels)
    val_ds = DigitImageDataset(val_images, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    # Fabric places/parallelizes DataLoaders as needed
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    return train_loader, val_loader


# ---------------------------
# Train & Eval for one epoch
# ---------------------------
def train_one_epoch(fabric, model, loader, criterion, optimizer):
    model.train()

    losses = []
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        optimizer.zero_grad()

        images, labels = fabric.to_device((images, labels))

        logits = model(images)
        loss = criterion(logits, labels)

        fabric.backward(loss)
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 10 == 0:
            acc = correct / max(total, 1)
            fabric.print(f"Step {step:04d} - train_loss={losses[-1]:.4f} - train_acc={acc:.4f}")

    epoch_acc = correct / max(total, 1)
    return losses, epoch_acc


@torch.no_grad()
def evaluate(fabric, model, loader, criterion):
    model.eval()

    losses = []
    correct = 0
    total = 0

    confmat_metric = MulticlassConfusionMatrix(num_classes=10).to(fabric.device)

    for images, labels in loader:
        images, labels = fabric.to_device((images, labels))
        logits = model(images)
        loss = criterion(logits, labels)

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        confmat_metric.update(preds, labels)

    val_acc = correct / max(total, 1)
    confmat = confmat_metric.compute().cpu()

    return losses, val_acc, confmat


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="../data/img_data/", show_default=True, help="Path to data directory.")
@click.option("--batch-size", type=int, default=64, show_default=True, help="Batch size for training.")
@click.option("--epochs", type=int, default=10, show_default=True, help="Number of training epochs.")
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Learning rate.")
@click.option("--weight-decay", type=float, default=1e-5, show_default=True, help="Weight decay (L2 penalty).")
@click.option("--ckpt", type=click.Path(file_okay=True), default="./cnn.ckpt", show_default=True, help="Path to save checkpoint.")
@click.option("--save-plot/--no-save-plot", default=False, show_default=True, help="Whether to plot requested curves.")
def main(data_dir, batch_size, epochs, lr, weight_decay, ckpt, save_plot):
    data_dir = Path(data_dir)
    ckpt_path = Path(ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42)  # reproducibility

    fabric = L.Fabric(accelerator="cpu")  # AG only CPU
    fabric.launch()

    train_loader, val_loader = prepare_dataloaders(data_dir, batch_size, fabric)

    model = SmallCNN(w=28, h=28, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model, optimizer = fabric.setup(model, optimizer)

    best_val_acc = 0.0
    best_confmat = None
    all_train_losses = []
    all_val_accs = []

    for epoch in range(1, epochs + 1):
        train_losses, train_acc = train_one_epoch(fabric, model, train_loader, criterion, optimizer)
        val_losses, val_acc, confmat = evaluate(fabric, model, val_loader, criterion)

        fabric.print(
            f"Epoch {epoch:02d} | "
            f"train_loss={sum(train_losses) / len(train_losses):.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={sum(val_losses) / len(val_losses):.4f}, val_acc={val_acc:.4f}"
        )
        all_train_losses.extend(train_losses)   # store per‑step losses
        all_val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_confmat = confmat
            fabric.print(f"New best val_acc={best_val_acc:.4f}, saving checkpoint to {ckpt_path}")
            fabric.save(ckpt_path, {"model": model.state_dict()})
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "model": model.state_dict(),
                },
                ckpt_path,
            )

    fabric.print(f"Best validation accuracy: {best_val_acc:.4f}")
    fabric.print("Confusion matrix for best checkpoint:")
    fabric.print(best_confmat)

    # Plotting (only if --save-plot is provided)
    if save_plot:
        import matplotlib.pyplot as plt

        # Loss vs. training step
        plt.figure()
        plt.plot(all_train_losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Loss vs. Training Step")
        plt.grid(True)
        plt.savefig("loss_vs_step.png")
        plt.close()

        # Validation accuracy vs. epoch
        plt.figure()
        plt.plot(range(1, len(all_val_accs) + 1), all_val_accs, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy vs. Epoch")
        plt.grid(True)
        plt.savefig("val_acc_vs_epoch.png")
        plt.close()

        fabric.print("Plots saved as loss_vs_step.png and val_acc_vs_epoch.png")


if __name__ == "__main__":
    main()