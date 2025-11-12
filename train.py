from torch.utils.data import DataLoader

from dataset import SignalsDataset


def main():
    batch_size = 10
    data_path = "samples.hdf5"

    dataset = SignalsDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    main()
