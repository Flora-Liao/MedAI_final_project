"""
augment.py
----------
Factory that converts a dict of flags → Albumentations Compose.
Example
-------
from augment import build_transforms
cfg = {"rotation":10, "h_flip":True, "clahe":{"clip":2.5}}
tfms = build_transforms(cfg)
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import functools

np.random.seed(42)   # reproducible

def add_speckle(img, noise_range, **kwargs):
    """
    Apply speckle noise to a uint8 H×W image.
    noise_range: (min_sigma, max_sigma)
    """
    img = img.astype(np.float32)
    sigma = np.random.uniform(*noise_range)
    gauss = np.random.randn(*img.shape).astype(np.float32)
    noisy = img + gauss * sigma * 255.0
    np.clip(noisy, 0, 255, out=noisy)
    return noisy.astype(np.uint8)


def build_transforms(cfg: dict, train: bool = True, totensor: bool = False) -> A.Compose:
    """
    Build an Albumentations pipeline from config flags.
    cfg keys:
      - rotation: int degrees for ± rotation
      - h_flip: bool
      - crop_resize: float scale lower bound
      - elastic: bool
      - speckle_noise: tuple(min_sigma, max_sigma)
    """
    t = []
    # 1) force every image to exactly 512×512
    #    (Pad up small ones, then Resize all – big or small – to 512×512)
    TGT = 256
    t.append(A.PadIfNeeded(
        min_height=TGT, min_width=TGT,
        border_mode=cv2.BORDER_CONSTANT, p=1.0
    ))
    t.append(A.Resize(height=TGT, width=TGT, p=1.0))

    # ── geometric transforms ────────────────────────────────────
    if deg := cfg.get("rotation"):
        t.append(A.Rotate(limit=deg, p=1.0))
    if cfg.get("h_flip", False):
        t.append(A.HorizontalFlip(p=0.5))
    if crop := cfg.get("crop_resize"):
        t.append(
            A.RandomResizedCrop(
                height=TGT,
                width=TGT,
                scale=(crop, 1.0),
                ratio=(0.9, 1.1),
                p=1.0
            )
        )

    if cfg.get("elastic", False):
        t.append(
            A.ElasticTransform(alpha=15, sigma=4, p=0.5)
        )

    # ── speckle noise ───────────────────────────────────────────
    if noise_range := cfg.get("speckle_noise"):
        # use a picklable partial so multiprocessing can serialize it
        speckle_fn = functools.partial(add_speckle, noise_range=noise_range)
        t.append(A.Lambda(image=speckle_fn))

    # ── to tensor ────────────────────────────────────────────────
    if totensor:
        t.append(ToTensorV2())

    return A.Compose(
        t,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # ← 因為我們轉進來的是 [x1, y1, x2, y2]
            label_fields=['class_labels']
        )
    )