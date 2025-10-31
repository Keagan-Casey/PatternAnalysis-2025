# dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # --- Load image ---
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Resize & normalize ---
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = torch.from_numpy(img).float() / 255.0

        # --- Load labels ---
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    c, x, y, bw, bh = map(float, line.strip().split())
                    boxes.append([c, x, y, bw, bh])
        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        sample = {
            "image": img,
            "bboxes": boxes,   # [class, x_center, y_center, w, h] (normalized)
            "img_path": str(img_path)
        }
        return sample


def get_loaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir, batch_size=8, img_size=640):
    train_dataset = YOLODataset(train_img_dir, train_label_dir, img_size)
    val_dataset = YOLODataset(val_img_dir, val_label_dir, img_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

