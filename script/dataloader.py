import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

MAX_LEN = 101


def load_csv_data(csv_path):
    return pd.read_csv(csv_path).fillna(0.0).values.astype(np.float32)


def load_sequences_and_labels(csv_path):
    df = pd.read_csv(csv_path, sep=r"[,\s]+", engine="python", header=None)
    sequences = df.iloc[:, 1].astype(str).str.upper().tolist()
    labels = (pd.to_numeric(df.iloc[:, 2]) > 0).astype(int).tolist()
    return sequences, labels


def load_multiple_shapes(shape_folder, prefix="Train"):
    shape_types = ["HelT", "MGW", "ProT", "Roll", "EP"]
    shape_list = [load_csv_data(os.path.join(shape_folder, f"{prefix}_{shape_type}.csv")) for shape_type in shape_types]
    return np.stack(shape_list, axis=1)


def load_conservation(cons_folder, prefix="Train"):
    data = load_csv_data(os.path.join(cons_folder, f"{prefix}_Conservation.csv"))
    return np.expand_dims(data, axis=1)


class MyLazyDataset(Data.Dataset):
    def __init__(self, sentences, labels, shapes, cons, mode="train"):
        self.sentences = sentences
        self.labels = labels
        self.shapes = shapes
        self.cons = cons

    def __len__(self):
        return len(self.sentences)

    def _to_onehot(self, seq):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        onehot = np.zeros((4, MAX_LEN), dtype=np.float32)
        for i, char in enumerate(seq[:MAX_LEN]):
            channel = mapping.get(char)
            if channel is not None:
                onehot[channel, i] = 1.0
        return torch.FloatTensor(onehot)

    def __getitem__(self, idx):
        seq_tensor = self._to_onehot(self.sentences[idx])
        shape = torch.FloatTensor(self.shapes[idx])
        con = torch.FloatTensor(self.cons[idx])
        label = self.labels[idx]
        return seq_tensor, shape, con, label


def my_custom_collate(batch):
    inputs = torch.stack([x[0] for x in batch])
    shapes = torch.stack([x[1] for x in batch])
    cons = torch.stack([x[2] for x in batch])
    labels = torch.tensor([x[3] for x in batch])
    return inputs, shapes, cons, labels
