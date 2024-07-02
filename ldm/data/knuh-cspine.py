import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KNUHCSpineBase(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 default_size=None,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 rgb_or_gray='rgb'
                 ):
        self.data_root = data_root
        self.split = split
        self.default_size = default_size
        self.rgb_or_gray = rgb_or_gray

        self.data = []
        self.labels = {}
        self.modes = ['A', 'B', 'C', 'D']
        self.label_to_idx = dict((mode, i) for i, mode in enumerate(self.modes))
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        class_label = []

        for label in os.listdir(f"{self.data_root}/{self.split}"):
            for img in os.listdir(f"{self.data_root}/{self.split}/{label}"):
                self.data.append(f"{self.data_root}/{self.split}/{label}/{img}")
                class_label.append(int(label))

        self.labels['class_label'] = np.array(class_label)
        self.labels['human_label'] = np.array([self.idx_to_label[x] for x in self.labels["class_label"]])

        self.size = size
        self.interpolation = {"linear": Image.Resampling.NEAREST,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = dict((k, self.labels[k][idx]) for k in self.labels)
        image = Image.open(self.data[idx])
        if self.rgb_or_gray.lower() == "rgb":
            image = image.convert("RGB")
        elif self.rgb_or_gray.lower() == "gray":
            image = image.convert("L")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class KNUHCSpineTrain(KNUHCSpineBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/KNUH_Cervical/knuh-cspine/images/", split='train', **kwargs)


class KNUHCSpineValidation(KNUHCSpineBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/KNUH_Cervical/knuh-cspine/images/", split='validation', **kwargs)


class KNUHCSpineTest(KNUHCSpineBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/KNUH_Cervical/knuh-cspine/images/", split='test', **kwargs)