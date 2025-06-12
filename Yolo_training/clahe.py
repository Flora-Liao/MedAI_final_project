import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

random.seed(22)

def extract_clahe_samples(dataset, cliplimit=2.0):
    """
    Args:
        dataset    : a Dataset or ConcatDataset (must have `.imgs` and `._read`)
        cliplimit  : float, CLAHE clipLimit parameter
        N          : number of samples to extract

    Returns:
        augmented:       list of original *augmented* image
        clahed_aug:      list of images after CLAHE
    """
    # build CLAHE operator once
    clahe_op = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
    augmented, clahed_aug = [], []

    for idx in range(len(dataset)):
        # 1) pull out whatever dataset[idx] gives you
        item = dataset[idx]
        # if dataset returns (img, label) or (img, mask, label), pick img at [0]
        if isinstance(item, (tuple, list)):
            img_t = item[0]
        else:
            img_t = item

        # 2) to numpy H×W in [0,1]
        if isinstance(img_t, torch.Tensor):
            img_np = img_t.detach().cpu().numpy()
        else:
            img_np = np.array(img_t)     # e.g. PIL → ndarray

        # if it’s C×H×W, squeeze down to H×W
        if img_np.ndim == 3:
            img_np = img_np.squeeze(0)

        augmented.append(img_np.astype(np.float32) / 255.0)

        # 3) convert to uint8 for CLAHE, apply it, then back to [0,1]
        img_uint8 = (img_np * 255).astype(np.uint8)
        cl = clahe_op.apply(img_uint8)
        clahed_aug.append(cl.astype(np.float32) / 255.0)

    return augmented, clahed_aug

# TODO: revise the code below
def show_clahe_results(originals, clahed, N=4):
    """
    Given two lists of images (float32 arrays in [0,1]), displays:
      - Col1: raw
      - Col2: CLAHE
      - Col3: raw histogram
      - Col4: CLAHE histogram
    """
    indices = random.sample(range(len(originals)), min(N, len(originals)))
    fig, axes = plt.subplots(N, 4, figsize=(12, 3*N))

    for i, idx in enumerate(indices):
        raw = originals[idx]
        cl = clahed[idx]

        ax = axes[i,0]
        ax.imshow(raw, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Raw");     ax.axis("off")

        ax = axes[i,1]
        ax.imshow(cl,  cmap="gray", vmin=0, vmax=1)
        ax.set_title("CLAHE");   ax.axis("off")

        ax = axes[i,2]
        ax.hist(raw.ravel(), bins=256, range=(0,1))
        ax.set_title("Raw Hist"); ax.set_xlim(0,1)

        ax = axes[i,3]
        ax.hist(cl.ravel(),  bins=256, range=(0,1))
        ax.set_title("CLAHE Hist"); ax.set_xlim(0,1)

    plt.tight_layout()
    plt.show()