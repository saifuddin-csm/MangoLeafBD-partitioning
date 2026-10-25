"""Image loading, augmentation and data loaders (Section 3.3.3, Image Preparation)."""
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ImageStore:
    """Optionally keeps decoded images in memory, removing JPEG decoding from every epoch."""
    def __init__(self, paths=(), cache=True):
        self.mem = {p: np.asarray(Image.open(p).convert("RGB")) for p in paths} if cache else {}

    def __call__(self, p):
        return Image.fromarray(self.mem[p]) if p in self.mem else Image.open(p).convert("RGB")


class PadSquare:
    """Aspect-ratio comparison (Table A4): pad to a square instead of squashing."""
    def __call__(self, im):
        w, h = im.size; s = max(w, h)
        bg = Image.new("RGB", (s, s), (124, 116, 104)); bg.paste(im, ((s - w) // 2, (s - h) // 2))
        return bg


def build_tf(train, img_size=224, resize="squash"):
    pre = [PadSquare()] if resize == "pad" else []
    pre += [transforms.Resize((img_size, img_size))]
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
           transforms.RandomRotation(15), transforms.ColorJitter(.15, .15, .15)] if train else []
    return transforms.Compose(pre + aug + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    def __init__(self, paths, labels, tf, store):
        self.p, self.y, self.tf, self.store = paths, labels, tf, store

    def __len__(self): return len(self.p)

    def __getitem__(self, i): return self.tf(self.store(self.p[i])), int(self.y[i])


def make_loaders(study, split, store, cfg, dev, resize="squash"):
    tr, va, te = split
    paths, labels = study.df.path.values, study.df.label.values

    def mk(ix, train, shuffle):
        ds = LeafDataset(paths[ix], labels[ix], build_tf(train, cfg.img_size, resize), store)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                          pin_memory=(dev == "cuda"), persistent_workers=False)
    return mk(tr, True, True), mk(va, False, False), mk(te, False, False)


def denorm(t):
    return np.clip(t.cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN), 0, 1)
