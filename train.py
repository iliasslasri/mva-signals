import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset import SignalsDataset
from dump_model import DumbSignalModel


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved at epoch {epoch+1} → {path}")


def validate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0

    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = loss_fn(outputs, labels.long())
            total_loss += loss.item() * signals.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

    avg_loss = total_loss / n
    accuracy = correct / n
    model.train()
    return avg_loss, accuracy


def main():
    batch_size = 300
    n_epochs = 1000
    train_path = "train.hdf5"
    val_path = "validation.hdf5"

    run_name = time.strftime("run_%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")

    # Load datasets
    train_dataset = SignalsDataset(train_path, "stft")
    val_dataset = SignalsDataset(val_path, "stft")

    model = DumbSignalModel(n_classes=6, n_channels=2)
    summary(model, input_data=train_dataset[0][0].unsqueeze(0))
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training Loop
    for epoch in range(n_epochs):
        for signals, labels, snr in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss_value = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # Compute validation metrics
        val_loss, val_acc = validate(model, val_loader, device, loss_fn)

        # Logging
        writer.add_scalar("Loss/train", loss_value.item(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if (epoch + 1) % 10 == 0:
            print(
                f"[{epoch+1}/{n_epochs}] "
                f"train_loss={loss_value.item():.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

    # Save final checkpoint
    save_checkpoint(model, optimizer, n_epochs - 1, loss_value.item(), path=ckpt_path)


if __name__ == "__main__":
    main()
