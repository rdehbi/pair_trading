import pkg_resources
import pandas as pd
import os

class Dataset:
    def __init__(self, path: str = None):
        if path is None:
            path = pkg_resources.resource_filename("pair_trading", "data")
        self.path = path

    @property
    def open(self):
        return pd.read_csv(os.path.join(self.path, "Open.csv"))

    @property
    def close(self):
        return pd.read_csv(os.path.join(self.path, "Close.csv"))

    @property
    def high(self):
        return pd.read_csv(os.path.join(self.path, "High.csv"))

    @property
    def low(self):
        return pd.read_csv(os.path.join(self.path, "Low.csv"))

    @property
    def volume(self):
        return pd.read_csv(os.path.join(self.path, "Volume.csv"))

    @property
    def stock_splits(self):
        return pd.read_csv(os.path.join(self.path, "Stock Splits.csv"))


class Simulator:
    def __init__(self, dataset):
        self.dataset = dataset

    def simulate(self, signal):
        raise NotImplementedError()  # todo


if __name__ == "__main__":
    dataset = Dataset()
    print(dataset.close)
