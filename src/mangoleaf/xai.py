"""Explainability analyses of the deployed model (Supplementary Methods): region ablation (Figure A2),
Grad-CAM and occlusion sensitivity (Figure 8)."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import SHORT
from .data import build_tf, denorm


class GradCAM:
    """Grad-CAM on the last feature map; token outputs of transformers are reshaped to a square grid."""
    def __init__(self, model, img_size=224):
        self.m, self.size = model.eval(), img_size

    @staticmethod
    def _map(t, m):
        if t.dim() == 4:
            _b, d1, d2, d3 = t.shape
            if d2 == d3 and d1 != d2: return t
            if d1 == d2 and d2 != d3: return t.permute(0, 3, 1, 2)
            return t
        if t.dim() == 3:
            npx = getattr(m, "num_prefix_tokens", 1); xx = t[:, npx:, :]
            n = xx.shape[1]; k = int(round(math.sqrt(n)))
            return None if k * k != n else xx.transpose(1, 2).reshape(xx.shape[0], -1, k, k)
        return None

    def __call__(self, x, cls):
        self.m.zero_grad(set_to_none=True)
        fe = self.m.forward_features(x); t = fe[0] if isinstance(fe, (tuple, list)) else fe
        t.retain_grad(); lg = self.m.forward_head(fe); lg[0, cls].backward()
        a, g = self._map(t, self.m), self._map(t.grad, self.m)
        if a is None or g is None: return None
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, (self.size, self.size), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min()) if cam.max() > cam.min() else cam


@torch.no_grad()
def occlusion(net, x, cls, size=224, patch=32, stride=16):
    """Drop in true-category probability when a 32-pixel patch is zeroed, at a stride of 16."""
    base = net(x).softmax(1)[0, cls].item(); pos = list(range(0, size - patch + 1, stride))
    co = [(a, b) for a in pos for b in pos]; h = np.zeros((len(pos), len(pos)), np.float32); var = []
    for a, b in co:
        v = x.clone(); v[:, :, a:a + patch, b:b + patch] = 0.; var.append(v[0])
    out = np.concatenate([net(torch.stack(var[i:i + 64])).softmax(1)[:, cls].cpu().numpy()
                          for i in range(0, len(var), 64)])
    for (a, b), pv in zip(co, out): h[pos.index(a), pos.index(b)] = base - pv
    h = np.clip(h, 0, None); h = h / h.max() if h.max() > 0 else h
    return F.interpolate(torch.tensor(h)[None, None], (size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy(), base


@torch.no_grad()
def region_ablation(net, study, test_ix, store, dev, fold, size=224):
    """True-category probability on the full image, the central half only, and with the centre removed."""
    etf = build_tf(False, size); rows = []
    lo, hi = size // 4, 3 * size // 4
    for ix in test_ix:
        img = etf(store(study.df.path.values[ix])).unsqueeze(0).to(dev); c = int(study.Y[ix])
        m1 = img.clone(); m1[:, :, lo:hi, lo:hi] = 0.
        m2 = torch.zeros_like(img); m2[:, :, lo:hi, lo:hi] = img[:, :, lo:hi, lo:hi]
        rows.append({"fold": fold, "category": study.classes[c], "full": net(img).softmax(1)[0, c].item(),
                     "centre_only": net(m2).softmax(1)[0, c].item(), "centre_removed": net(m1).softmax(1)[0, c].item()})
    return pd.DataFrame(rows)


def fig8_gradcam_occlusion(net, study, test_ix, store, dev, rng, size=224):
    """Figure 8: one test leaf per category from sessions unseen in training."""
    etf = build_tf(False, size); cg = GradCAM(net, size); Y = study.Y; nc = study.num_classes
    pick = [int(rng.choice(test_ix[Y[test_ix] == c])) for c in range(nc)]
    fig, axes = plt.subplots(3, nc, figsize=(1.0 * nc, 3.25))
    for j, ix in enumerate(pick):
        img = etf(store(study.df.path.values[ix])); x = img.unsqueeze(0).to(dev); c = int(Y[ix])
        with torch.enable_grad():
            cam = cg(x, c)
        om, base = occlusion(net, x, c, size)
        for r in range(3): axes[r, j].imshow(denorm(img))
        if cam is not None: axes[1, j].imshow(cam, cmap="jet", alpha=.45)
        axes[2, j].imshow(om, cmap="jet", alpha=.45)
        axes[0, j].set_title(f"{SHORT[study.classes[c]]}\np = {base:.2f}", fontsize=6.5, pad=2)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
    for r, lab in enumerate(("Input", "Grad-CAM", "Occlusion")): axes[r, 0].set_ylabel(lab, fontsize=7)
    plt.subplots_adjust(wspace=0.04, hspace=0.06)
    return fig
