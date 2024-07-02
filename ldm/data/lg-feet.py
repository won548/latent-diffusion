import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LGFeetBase(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 default_size=None,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 rgb_or_gray='rgb',
                 views=None
                 ):
        self.data_root = data_root
        self.split = split
        self.default_size = default_size
        self.rgb_or_gray = rgb_or_gray
        self.views = views

        self.data = []
        if views is None:
            views = ['axial', 'coronal', 'sagittal']

        for view in views:
            for img in os.listdir(f"{self.data_root}/{self.split}/image/{view}"):
                self.data.append(f"{self.data_root}/{self.split}/image/{view}/{img}")

        self.size = size
        self.interpolation = {"linear": Image.Resampling.NEAREST,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]

        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=flip_p),
                                             transforms.RandomVerticalFlip(p=flip_p)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = {}
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

        if self.split == 'train':
            image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class LGFeetTrain(LGFeetBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='train', **kwargs)


class LGFeetValidation(LGFeetBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='validation', **kwargs)


class LGFeetTest(LGFeetBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='test', **kwargs)
