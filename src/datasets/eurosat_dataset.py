from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


class EurosatDataset(Dataset):

    def __init__(self, root, split, transform=None, get_path=False):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        with open(self.root / f'{split}.txt') as f:
            filenames = f.read().splitlines()

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.get_path = get_path

        self.samples = []
        for fn in filenames:
            cls_name = fn.split('_')[0]
            self.samples.append(self.root / cls_name / fn)

    def __getitem__(self, index):
        path = self.samples[index]

        img = Image.open(path)
        target = self.class_to_idx[path.parts[-2]]

        if self.transform is not None:
            img = self.transform(img)

        if self.get_path:
            return img, target, str(path)
        else:
            return img, target

    def __len__(self):
        return len(self.samples)
