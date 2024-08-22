"""
For training from a pre-embedded dataset of image-text pairs from a clip model


Make sure that the CLIP model you use is the same as the one used to obtain the
pre embeddings
"""

import torch.utils.data
from math import ceil
from typing import List
import lightning.pytorch as pl
import numpy as np
from glob import glob
import re
import os
import six
import lmdb
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
import pickle

def get_file_code(filename: str):
    return os.path.basename(filename).split(".")[-2]


def random_sort(*lists):
    indices = np.random.permutation(len(lists[0]))
    return [l[i] for i in indices for l in lists]


class MinecraftDataset(Dataset):
    def __init__(self, path: str):
        self.img_files = sorted(glob(os.path.join(path, "*/*.npy")))

        assert len(self.img_files) > 0
        print("Found", len(self.img_files), "files")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_dat = np.load(self.img_files[idx])
        return img_dat


class MinecraftIterableDataset(IterableDataset):
    def __init__(self, path: str):
        self.img_files = sorted(glob(os.path.join(path, "*/*.npy")))

        assert len(self.img_files) > 0, "No files found"
        print(f"Found {len(self.img_files)} files")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_start = 0
            iter_end = len(self.img_files)
        else:  # in a worker process
            # Split dataset among workers
            per_worker = int(ceil(len(self.img_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.img_files))

        for idx in range(iter_start, iter_end):
            yield self.process_file(self.img_files[idx])

    def process_file(self, file_path):
        return np.load(file_path)


class MinecraftDatasetLMDB(Dataset):
    def __init__(self, path: str, lmdb_path: str):
        self.clip_files = sorted(glob(os.path.join(path, "*.npy")))

        assert len(self.clip_files) > 0
        print("Found", len(self.clip_files), "files")

        self.lmdb = None
        self.lmdb_out_path = lmdb_path

        self.indices_path  = os.path.join(self.lmdb_out_path, "indices.pkl")
        self.indices = None
        if os.path.exists(self.indices_path):
            self.indices = pickle.load(open(self.indices, "rb"))

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.lmdb is None:
            self._init_lmdb()
        
        clip_path, data_idx = self.indices[idx]
        clip_buf = self.lmdb.begin(write=False).get(clip_path.encode())
        buf = six.BytesIO()
        buf.write(clip_buf)
        buf.seek(0)
        data = np.load(buf)
        
        return data[data_idx]

    def _init_lmdb(self):
        self.lmdb = lmdb.open(self.lmdb_out_path, readonly=True, lock=False, readahead=True, meminit=False)
    
    def create_lmdb_database(self):
        if os.path.exist(self.lmdb_out_path):
            print("LMDB database already exists. Not generating new LMDB datasbse")
        
        file_lens = []
        indices = []
        with lmdb.open(self.lmdb_out_path, map_size=220*1e9).begin(write=True) as txn:
            for img_file in tqdm(self.clip_files):
                clip_np = np.load(img_file)
                file_lens.append(len(clip_np))
                indices.extend([(img_file, i) for i in range(len(clip_np))])

                raw_clip_bytes = clip_np.tobytes()
                txn.put(img_file.encode(), raw_clip_bytes)
            txn.commit()
        
        pickle.dump(indices, open(self.indices_path, "wb"))
        print("LMDB database created at", self.lmdb_out_path)
        print(f"Wrote {len(self.clip_files)} files")


class MinecraftEmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_train: str,
        path_val: str,
        batch_size: int,
        *_,
        **kwargs
    ):
        """
        path_train: some path to a directory with the following subfolders:
            images/*.npy
            texts/*.npy

        This module will iterate over all rows of all npy files.
        """
        super().__init__()
        self.batch_size = batch_size

        # self.ds_train = IterableMinecraftDataset(path_train, batch_size, )
        # self.ds_test = IterableMinecraftDataset(path_val, batch_size, )

        self.ds_train = MinecraftDatasetLMDB(path_train)
        self.ds_test = MinecraftDatasetLMDB(path_val)

    def train_dataloader(self):
        return DataLoader(self.ds_train, num_workers=32, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.ds_test, num_workers=4, batch_size=self.batch_size, shuffle=False)
    
if __name__ == "__main__":
    ds_train = MinecraftDatasetLMDB("/cvdata1/jihwan/data/minecraft_clip/train", "/cvdata1/jihwan/data/minecraft_clip/lmdb/train.lmdb")
    ds_train.create_lmdb_database()
    ds_val = MinecraftDatasetLMDB("/cvdata1/jihwan/data/minecraft_clip/test", "/cvdata1/jihwan/data/minecraft_clip/lmdb/test.lmdb")
    ds_val.create_lmdb_database()
