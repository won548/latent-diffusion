import os

import cv2
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

        # Initialize accumulators for each class
        self.class_accumulators = {mode: None for mode in self.modes}
        self.class_counts = {mode: 0 for mode in self.modes}

        for label in os.listdir(f"{self.data_root}/{self.split}"):
            for img in os.listdir(f"{self.data_root}/{self.split}/{label}"):
                img_path = f"{self.data_root}/{self.split}/{label}/{img}"
                self.data.append(img_path)
                class_label.append(int(label))

                # Load image
                image = Image.open(img_path)
                if self.rgb_or_gray.lower() == "rgb":
                    image = image.convert("RGB")
                elif self.rgb_or_gray.lower() == "gray":
                    image = image.convert("L")
                img_array = np.array(image).astype(np.float32)

                # Accumulate the sum for each class
                mode = self.idx_to_label[int(label)]
                if self.class_accumulators[mode] is None:
                    self.class_accumulators[mode] = np.zeros_like(img_array)
                self.class_accumulators[mode] += img_array
                self.class_counts[mode] += 1

        self.labels['class_label'] = np.array(class_label)
        self.labels['human_label'] = np.array([self.idx_to_label[x] for x in self.labels["class_label"]])

        self.size = size
        self.interpolation = {"linear": Image.Resampling.NEAREST,
                              "bilinear": Image.Resampling.BILINEAR,
                              "bicubic": Image.Resampling.BICUBIC,
                              "lanczos": Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

        # Calculate mean image for each class
        self.class_mean_images = {}
        for mode in self.modes:
            if self.class_counts[mode] > 0:
                self.class_mean_images[mode] = (self.class_accumulators[mode] / self.class_counts[mode]).astype(
                    np.float32)
                mean_image = self.class_mean_images[mode].astype(np.uint8)
                os.makedirs('samples/knuh-cspine-mean', exist_ok=True)
                cv2.imwrite(f'samples/knuh-cspine-mean/{mode}.png', mean_image)
            else:
                self.class_mean_images[mode] = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = dict((k, self.labels[k][idx]) for k in self.labels)
        image = Image.open(self.data[idx])
        if self.rgb_or_gray.lower() == "rgb":
            image = image.convert("RGB")
        elif self.rgb_or_gray.lower() == "gray":
            image = image.convert("L")

        # Default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Include mean image for the corresponding class
        class_label = example['class_label']
        mode = self.idx_to_label[class_label]
        mean_image = self.class_mean_images[mode]

        # Resize mean image if necessary and add to example
        if mean_image is not None and self.size is not None:
            mean_image_pil = Image.fromarray(mean_image.astype(np.uint8))
            mean_image_pil = mean_image_pil.resize((self.size, self.size), resample=self.interpolation)

        if np.random.rand(1) < self.flip_p:
            image = self.flip(image)
            mean_image_pil = self.flip(mean_image_pil)

        image = np.array(image).astype(np.uint8)
        mean_image = np.array(mean_image_pil).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["image_mean"] = (mean_image / 127.5 - 1.0).astype(np.float32) if mean_image is not None else None

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