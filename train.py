import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from dataset import SignalsDataset
from DSFT_model import DSFTSignalModel
from dump_model import DumbSignalModel
from utils import count_n_param


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
    batch_size = 128
    n_epochs = 1000
    train_path = "train.hdf5"
    val_path = "validation.hdf5"

    # ----------------------------
    # Parse CLI arguments
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default=None, help="Optional name added to the run folder"
    )
    args = parser.parse_args()

    # ----------------------------
    # Build run name
    # ----------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if args.name is None:
        final_run_name = f"run_{timestamp}"
    else:
        final_run_name = f"{args.name}_{timestamp}"

    log_dir = os.path.join("runs", final_run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    ckpt_path = os.path.join(log_dir, "checkpoint.pt")
    print(f"Logging to {log_dir}")

    # Load datasets
    train_dataset = SignalsDataset(train_path, "stft")
    val_dataset = SignalsDataset(val_path, "stft")

    model = DSFTSignalModel(n_classes=6, n_channels=2)
    summary(model, input_data=torch.zeros((16, 2, 7, 7)))
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training Loop
    for epoch in range(n_epochs):
        step = 0
        for signals, labels, snr in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss_value = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss/train_step", loss_value.item(), step + epoch * len(train_loader)
            )
            step += 1

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
    main()
    main()
