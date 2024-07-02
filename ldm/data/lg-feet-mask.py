import os

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resize_channel(channel, new_width, new_height):
    return cv2.resize(channel, (new_width, new_height))


class LGFeetMaskBase(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 default_size=None,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 rgb_or_gray='rgb',
                 views=None,
                 use_mask=False
                 ):
        self.data_root = data_root
        self.split = split
        self.default_size = default_size
        self.rgb_or_gray = rgb_or_gray
        self.views = views
        self.use_mask = use_mask

        self.data = []
        if views is None:
            views = ['axial', 'coronal', 'sagittal']

        for view in views:
            for img in os.listdir(f"{self.data_root}/{self.split}/image/{view}"):
                self.data.append(f"{self.data_root}/{self.split}/image/{view}/{img}")
            self.data = sorted(self.data)

        if use_mask is True:
            self.data_mask = []
            if views is None:
                views = ['axial', 'coronal', 'sagittal']

            for view in views:
                for img_mask in os.listdir(f"{self.data_root}/{self.split}/mask/{view}"):
                    self.data_mask.append(f"{self.data_root}/{self.split}/mask/{view}/{img_mask}")
            self.data_mask = sorted(self.data_mask)

        self.size = size
        self.interpolation = {"linear": Image.Resampling.NEAREST,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]

        if split == 'train':
            self.transforms = A.Compose(transforms=[A.Resize(width=256, height=256),
                                                    A.HorizontalFlip(p=flip_p),
                                                    A.VerticalFlip(p=flip_p)], is_check_shapes=False)
        else:
            self.transforms = A.Compose(transforms=[A.Resize(width=256, height=256)], is_check_shapes=False)

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

        if self.use_mask is True:
            masks = []
            mask = np.load(self.data_mask[idx])
            mask = np.array(mask).astype(np.uint8)
            for ch in range(mask.shape[-1]):
                masks.append(mask[:, :, ch])

        if self.use_mask is True:
            transformed = self.transforms(image=img, masks=masks)
            image, masks = transformed["image"], transformed["masks"]
            masks = np.array(masks).astype(np.uint8)
            masks = np.transpose(masks, (1, 2, 0))
            example["mask"] = masks
        else:
            image = self.transforms(img)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


class LGFeetTrain(LGFeetMaskBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='train', **kwargs)


class LGFeetValidation(LGFeetMaskBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='validation', **kwargs)


class LGFeetTest(LGFeetMaskBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/media/NAS/nas_32/dongkyu/Datasets/LG_Feet_PNG", split='test', **kwargs)
