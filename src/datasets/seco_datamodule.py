from pytorch_lightning import LightningDataModule
from pl_bolts.models.self_supervised.moco.transforms import Moco2TrainImagenetTransforms

from datasets.seco_dataset import *
from utils.data import InfiniteDataLoader
from torch.utils.data import DataLoader
from utils.transforms import ApplyN
from torch.utils.data.sampler import Sampler
from random import shuffle

class SeasonalContrastBaseDataModule(LightningDataModule):

    def __init__(self, data_dir, bands=None, batch_size=32, num_workers=16, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None

    def setup(self, stage=None):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        self.train_dataset = self.get_dataset(root=self.data_dir, bands=self.bands, transform=train_transforms)

    @staticmethod
    def get_dataset(root, bands, transform):
        raise NotImplementedError

    @staticmethod
    def train_transform():
        raise NotImplementedError

    def train_dataloader(self):
        # return InfiniteDataLoader(
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True
        )


class SeasonalContrastBasicDataModule(SeasonalContrastBaseDataModule):
    num_keys = 1

    @staticmethod
    def get_dataset(root, bands, transform):
        return SeasonalContrastBasic(root, bands, transform)

    @staticmethod
    def train_transform():
        transform = Moco2TrainImagenetTransforms(height=224).train_transform
        return ApplyN(transform=transform, n=2)


class SeasonalContrastTemporalDataModule(SeasonalContrastBaseDataModule):
    num_keys = 1

    @staticmethod
    def get_dataset(root, bands, transform):
        return SeasonalContrastTemporal(root, bands, transform)

    @staticmethod
    def train_transform():
        return Moco2TrainImagenetTransforms(height=224).train_transform


class SeasonalContrastMultiAugDataModule(SeasonalContrastBaseDataModule):
    num_keys = 3

    @staticmethod
    def get_dataset(root, bands, transform):
        return SeasonalContrastMultiAug(root, bands, transform)

    @staticmethod
    def train_transform():
        return None

class FirstSecondSampler(Sampler):
# class FirstSecondSampler(DistributedSampler):
    def __init__(self, data):
        self.num_samples = len(data)//2
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     self.world_size = torch.distributed.get_world_size()
        #     self.rank = torch.distributed.get_rank()

    def __iter__(self):
        #FIX
        # lis = list(range((self.num_samples//self.world_size)*self.rank, (self.num_samples//self.world_size)*(self.rank+1)))
        lis = list(range(self.num_samples))
        shuffle(lis)
        lisfs = []
        for ele in lis:
            lisfs.append(ele*2)
            lisfs.append(ele*2+1)
        return iter(lisfs)

    def __len__(self):
        #FIX
        # return self.num_samples*2//self.world_size
        return self.num_samples*2

class TemporalContrastMultiAugDataModule(SeasonalContrastBaseDataModule):
    num_keys = 3

    @staticmethod
    def get_dataset(root, bands, transform):
        return TemporalContrastMultiAug(root, bands, transform)

    @staticmethod
    def train_transform():
        return None

    def train_dataloader(self):
        sampler = FirstSecondSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=sampler,
            persistent_workers=False
        )

class ChangeAwareContrastMultiAugDataModule(SeasonalContrastBaseDataModule):
    num_keys = 3

    @staticmethod
    def get_dataset(root, bands, transform):
        return ChangeAwareContrastMultiAug(root, bands, transform)

    @staticmethod
    def train_transform():
        return None

    def train_dataloader(self):
        sampler = FirstSecondSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=sampler,
            persistent_workers=False
        )