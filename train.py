import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchsummary import summary as tsum

from dataset import SignalsDataset
from dump_model import DumbSignalModel


def main():
    batch_size = 10
    n_epochs = 1000
    data_path = "train.hdf5"

    dataset = SignalsDataset(data_path)
    model = DumbSignalModel(n_classes=6, n_channels=2)
    summary(model, input_data=dataset[0][0].unsqueeze(0))
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate through one epoch
    for epoch in range(n_epochs):
        for _, (signals, labels, snr) in enumerate(dataloader):
            signals = signals.to(device)  # [B, C, T]
            labels = labels.to(device)  # [B]
            snr = snr.to(device)  # [B]
            # Forward pass
            outputs = model(signals)  # [B, 6]

            # Compute loss
            loss_value = loss(outputs, labels.long())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_value.item():.4f}")


if __name__ == "__main__":
    main()
