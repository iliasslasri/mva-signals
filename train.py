import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from dataset import SignalsDataset
from dump_model import DumbSignalModel


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pt"):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved at epoch {epoch+1} at {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        last_loss = ckpt["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}, loss={last_loss:.4f}")
        return start_epoch, last_loss
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, None


def main():
    batch_size = 300
    n_epochs = 1000
    data_path = "train.hdf5"

    dataset = SignalsDataset(data_path)
    model = DumbSignalModel(n_classes=6, n_channels=2)
    summary(model, input_data=dataset[0][0].unsqueeze(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/signal_training")

    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for _, (signals, labels, snr) in enumerate(dataloader):
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss_value = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # Update batch loss
            writer.add_scalar("Loss/batch", loss_value.item(), global_step)
            global_step += 1

            epoch_loss += loss_value.item()

        epoch_loss /= len(dataloader)

        # Log epoch loss
        writer.add_scalar("Loss/epoch", epoch_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        n_epochs - 1,
        epoch_loss,
        path=f"checkpoint_{n_epochs}_epochs.pt",
    )

    writer.close()


if __name__ == "__main__":
    main()
