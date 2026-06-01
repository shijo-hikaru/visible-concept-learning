import math
import os
import random
from collections import defaultdict
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import Flowers102, OxfordIIITPet, StanfordCars


class UnifiedSeenUnseenDataset:
    """
    必須:
      self.labels      : List[int]
      self.get_item(i) : (img, label, filename)
    """

    def _build_indices(
        self,
        mode="seen",
        seen_classes=None,
        seed=0,
        n_shot=None,
    ):
        assert mode in ["seen", "unseen", "all"]

        all_classes = sorted(set(self.labels))

        if mode != "all":
            if seen_classes is None:
                n_seen = len(all_classes) // 2
                seen_classes = all_classes[:n_seen]
                unseen_classes = all_classes[n_seen:]
            else:
                if seen_classes:
                    unseen_classes = set(all_classes) - set(seen_classes)

            self.seen_classes = set(seen_classes)
            self.unseen_classes = set(unseen_classes)

            if mode == "seen":
                use_classes = self.seen_classes
            elif mode == "unseen":
                use_classes = self.unseen_classes

        else:
            self.seen_classes = set(all_classes)
            self.unseen_classes = set()
            use_classes = set(all_classes)

        class_to_indices = defaultdict(list)
        for i, y in enumerate(self.labels):
            if y in use_classes:
                class_to_indices[y].append(i)

        rng = random.Random(seed)
        indices = []
        for y, idxs in class_to_indices.items():
            if n_shot is not None:
                rng.shuffle(idxs)
                idxs = idxs[:n_shot]
            indices.extend(idxs)

        return sorted(indices)


class PetWithFilename(OxfordIIITPet, UnifiedSeenUnseenDataset):
    def __init__(
        self,
        root,
        split="trainval",
        transform=None,
        download=False,
        mode="seen",
        seen_classes=None,
        seed=0,
        n_shot=None,
        remap_labels=False,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )

        self.labels = self._labels

        self.indices = self._build_indices(
            mode, seen_classes, seed, n_shot
        )

        self.remap_labels = remap_labels
        classes = sorted(self.seen_classes if mode == "seen" else
                         self.unseen_classes if mode == "unseen"
                         else set(self.labels))
        self.label_map = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = Image.open(self._images[real_idx]).convert("RGB")
        label = self.labels[real_idx]

        if self.remap_labels:
            label = self.label_map[label]

        if self.transform:
            img = self.transform(img)

        full_path = str(self._images[real_idx])
        filename = os.path.basename(self._images[real_idx])

        return img, label, filename, full_path


class CUBWithFilename(datasets.ImageFolder, UnifiedSeenUnseenDataset):
    def __init__(
        self,
        root,
        transform=None,
        mode="seen",
        seen_classes=None,
        seed=0,
        n_shot=None,
        remap_labels=False,
    ):
        super().__init__(root, transform=None)

        self.base_transform = transform

        self.labels = [y for _, y in self.samples]

        self.indices = self._build_indices(
            mode=mode,
            seen_classes=seen_classes,
            seed=seed,
            n_shot=n_shot,
        )

        self.remap_labels = remap_labels
        classes = sorted(self.seen_classes if mode == "seen" else
                         self.unseen_classes if mode == "unseen"
                         else set(self.labels))
        self.label_map = {c: i for i, c in enumerate(classes)}

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        try:
            classes.sort(key=lambda x: int(x))
        except ValueError:
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        path, label = self.samples[real_idx]

        if self.remap_labels:
            label = self.label_map[label]

        img = Image.open(path).convert("RGB")

        if self.base_transform:
            img = self.base_transform(img)

        filename = os.path.basename(path)

        return img, label, filename, str(path)

class AugmentedDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        valid_files=None,
        seen_classes=None,
        remap_labels=False,
    ):
        """
        root/
          ├── 0/
          │    ├── xxx.jpg
          ├── 1/
          │    ├── yyy.jpg
        """
        self.root = root
        self.transform = transform
        self.samples = []
        self.valid_files = valid_files
        self.remap_labels = remap_labels

        if seen_classes is not None:
            seen_classes = set(seen_classes)

        for class_id in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_id)

            if not os.path.isdir(class_dir):
                continue

            class_id_int = int(class_id)

            if seen_classes is not None and class_id_int not in seen_classes:
                continue

            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(('.jpg', '.png')):
                    continue

                if valid_files is not None and fname not in valid_files:
                    continue

                path = os.path.join(class_dir, fname)
                self.samples.append((path, class_id_int))

        used_labels = sorted({label for _, label in self.samples})
        self.label_map = {c: i for i, c in enumerate(used_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")
        filename = os.path.basename(path)

        if self.remap_labels:
            label = self.label_map[label]

        if self.transform:
            img = self.transform(img)

        return img, label, filename


class CUBDataset_train(datasets.ImageFolder):
    def __init__(self, root, transform=None, path2parts=None, is_train=True, include=None, exclude=None):
        self.base_transform = transform
        self.include = include
        self.exclude = exclude
        self.is_train = is_train
        self.path2parts = path2parts
        self.part_mapping = {
            'crown': 'head', 'forehead': 'head',
            'left eye': 'eyes', 'right eye': 'eyes',
            'beak': 'beak',
            'nape': 'neck/throat', 'throat': 'neck/throat',
            'breast': 'breast', 'belly': 'belly', 'back': 'back',
            'left wing': 'wings', 'right wing': 'wings',
            'left leg': 'legs', 'right leg': 'legs',
            'tail': 'tail',
        }
        self.mapped_parts = ['head', 'eyes', 'beak', 'neck/throat', 'breast', 'belly', 'back', 'wings', 'legs', 'tail']
        super().__init__(root, transform=None)

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        if self.include:
            classes = [c for c in classes if c in self.include]
        try:
            classes.sort(key=lambda x: int(x))
        except ValueError:
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]

        parts = self.path2parts[path.split('images/')[-1]]
        if self.is_train and random.random() < 0.6:
            if random.random() <= 1.0:
                image, parts = bold_crop_with_visibility(image, parts, crop_ratio=(0.3, 0.5), focus_on_part=True if random.random() < 0.5 else False)
            else:
                image, parts = part_mask_group(image, parts, mask_size=50)

        if self.base_transform:
            image = self.base_transform(image)

        visible_parts = []
        for x in parts:
            if parts[x][-1] == 1:
                visible_parts.append(self.part_mapping[x])
        visible_parts = list(set(visible_parts))
        visible_parts = [True if p in visible_parts else False for p in self.mapped_parts]
        return (image, label) + (path, torch.tensor(visible_parts))


def bold_crop_with_visibility(img, part_coords, crop_ratio=(0.3, 0.6), min_visible_parts=1, focus_on_part=False):
    W, H = img.size
    for _ in range(10):
        scale = random.uniform(*crop_ratio)
        new_W, new_H = int(W * scale), int(H * scale)
        if focus_on_part:
            visible_parts = [(name, (x, y)) for name, (x, y, v) in part_coords.items() if v == 1]
            if not visible_parts:
                return img, part_coords
            _, (px, py) = random.choice(visible_parts)
            left = max(0, int(px - new_W / 2))
            top = max(0, int(py - new_H / 2))
            right = min(W, left + new_W)
            bottom = min(H, top + new_H)
        else:
            left = random.randint(0, W - new_W)
            top = random.randint(0, H - new_H)
            right, bottom = left + new_W, top + new_H
        visible_count = sum(1 for (x, y, v) in part_coords.values() if v == 1 and left <= x <= right and top <= y <= bottom)
        if visible_count >= min_visible_parts:
            cropped = img.crop((left, top, right, bottom))
            new_parts = {}
            for name, (x, y, v) in part_coords.items():
                if v == 1 and left <= x <= right and top <= y <= bottom:
                    new_parts[name] = [x - left, y - top, 1]
                else:
                    new_parts[name] = [0.0, 0.0, 0]
            return cropped, new_parts
    return img, part_coords


def part_mask_group(img, part_coords, mask_size=50):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    new_parts = part_coords.copy()
    visible_parts = [(name, x, y) for name, (x, y, v) in part_coords.items() if v == 1]
    if not visible_parts:
        return img, part_coords
    target_name, tx, ty = random.choice(visible_parts)
    to_mask = [(name, x, y) for name, (x, y, v) in part_coords.items()
               if v == 1 and math.sqrt((tx - x) ** 2 + (ty - y) ** 2) <= mask_size]
    fill_color = (0, 0, 0) if random.random() < 0.5 else (128, 128, 128)
    for name, x, y in to_mask:
        left = max(0, int(x - mask_size // 2))
        top = max(0, int(y - mask_size // 2))
        right = min(img.width, int(x + mask_size // 2))
        bottom = min(img.height, int(y + mask_size // 2))
        draw.rectangle([left, top, right, bottom], fill=fill_color)
        new_parts[name] = [0.0, 0.0, 0]
    return img, new_parts


class Flowers102WithFilename(Flowers102, UnifiedSeenUnseenDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        download=False,
        mode="seen",
        seen_classes=None,
        seed=0,
        n_shot=None,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )

        self.labels = self._labels
        self.indices = self._build_indices(mode, seen_classes, seed, n_shot)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = super().__getitem__(real_idx)
        filename = os.path.basename(self._image_files[real_idx])
        return img, label, filename


class StanfordCarsWithFilename(StanfordCars, UnifiedSeenUnseenDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        download=False,
        mode="seen",
        seen_classes=None,
        seed=0,
        n_shot=None,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )
        self.labels = [label for _, label in self._samples]
        self.indices = self._build_indices(mode=mode, seen_classes=seen_classes, seed=seed, n_shot=n_shot)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = super().__getitem__(real_idx)
        path = self._samples[real_idx][0]
        filename = os.path.basename(path)
        return img, label, filename
