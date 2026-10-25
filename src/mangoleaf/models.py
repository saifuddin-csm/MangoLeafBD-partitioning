"""Model creation, fine-tuning (Section 3.5), test-time augmentation (Section 3.6.1), checkpoints and
inference cost (Table 9)."""
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from .data import make_loaders
from .utils import set_seed


class ModelZoo:
    def __init__(self, cfg, num_classes, dev):
        self.cfg, self.nc, self.dev = cfg, num_classes, dev
        self.amp = dev == "cuda"
        self.params = {}

    def _file(self, name):
        return self.cfg.pretrained / f"{name}_c{self.nc}.pt"

    def fetch(self, names, tries=4, base=15):
        """Download ImageNet weights once (timm), replace the head, and cache the state dict."""
        for n in sorted(set(names)):
            f = self._file(n)
            if not f.exists():
                last = None
                for i in range(tries):
                    try:
                        m = timm.create_model(n, pretrained=not self.cfg.dry_run, num_classes=self.nc); break
                    except Exception as e:
                        last = e; print(f"   retry {n} in {base * 2 ** i}s ({type(e).__name__})")
                        time.sleep(base * 2 ** i)
                else:
                    raise RuntimeError(f"could not load {n}") from last
                torch.save(m.state_dict(), f); del m
            sd = torch.load(f, map_location="cpu")
            self.params[n] = int(sum(v.numel() for k, v in sd.items()
                                     if v.dtype.is_floating_point and "running" not in k))
            print(f"  {n:32s} {self.params[n] / 1e6:6.1f} M parameters")
        os.environ["HF_HUB_OFFLINE"] = "1"

    def build(self, name):
        m = timm.create_model(name, pretrained=False, num_classes=self.nc)
        m.load_state_dict(torch.load(self._file(name), map_location="cpu"))
        return m.to(self.dev)

    @torch.no_grad()
    def predict(self, m, dl, tta):
        """Softmax probabilities; with tta, averaged over the identity and three reflections."""
        m.eval(); P, Yo = [], []
        for x, y in dl:
            x = x.to(self.dev, non_blocking=True)
            views = [x] + ([torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])] if tta else [])
            acc = 0
            for v in views:
                with torch.autocast("cuda", enabled=self.amp):
                    acc = acc + m(v).float().softmax(1)
            P.append((acc / len(views)).cpu().numpy()); Yo.append(y.numpy())
        return np.concatenate(P), np.concatenate(Yo)

    def train_one(self, name, study, split, store, seed, lr=None, resize="squash"):
        """AdamW, one-cycle schedule, label smoothing, mixed precision; the epoch with the highest validation
        accuracy is retained."""
        c = self.cfg; lr = c.lr if lr is None else lr
        set_seed(seed); m = self.build(name)
        tl, vl, sl = make_loaders(study, split, store, c, self.dev, resize)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=c.weight_decay)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tl), epochs=c.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        crit = nn.CrossEntropyLoss(label_smoothing=c.label_smooth)
        best, best_ep, bstate, hist = -1.0, 0, None, []
        for ep in range(c.epochs):
            m.train(); run, t0 = 0.0, time.time()
            for x, y in tl:
                x, y = x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=self.amp):
                    loss = crit(m(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
                run += loss.item() * x.size(0)
            pv, yv = self.predict(m, vl, tta=False); va = float(accuracy_score(yv, pv.argmax(1)))
            hist.append({"epoch": ep + 1, "train_loss": run / len(tl.dataset), "val_acc": va})
            print(f"    ep {ep + 1}/{c.epochs} loss {run / len(tl.dataset):.4f} val {va:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if va > best:
                best, best_ep = va, ep + 1
                bstate = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bstate)
        out = {"P_test": self.predict(m, sl, tta=True)[0], "P_test_plain": self.predict(m, sl, tta=False)[0],
               "P_val": self.predict(m, vl, tta=True)[0], "best_epoch": best_ep, "val_acc": best, "hist": hist}
        return m, out

    # ---- checkpoints (kept only for the first SW partition, used by the explainability analyses)
    def save_ckpt(self, m, run_id):
        sd = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in m.state_dict().items()}
        torch.save(sd, self.cfg.ckpt / (run_id.replace("|", "__") + ".pt"))

    def has_ckpt(self, run_id):
        return (self.cfg.ckpt / (run_id.replace("|", "__") + ".pt")).exists()

    def load_ckpt(self, name, run_id):
        m = timm.create_model(name, pretrained=False, num_classes=self.nc)
        sd = torch.load(self.cfg.ckpt / (run_id.replace("|", "__") + ".pt"), map_location="cpu")
        m.load_state_dict({k: (v.float() if v.dtype.is_floating_point else v) for k, v in sd.items()})
        return m.to(self.dev).eval()

    def latency(self, names):
        """Single-image fp32 inference latency (median of 100 runs after 10 warm-up runs) and model size."""
        rows = []
        for n in names:
            m = self.build(n).eval(); x = torch.randn(1, 3, self.cfg.img_size, self.cfg.img_size, device=self.dev)
            reps = 100 if self.dev == "cuda" else 5
            with torch.no_grad():
                for _ in range(10 if self.dev == "cuda" else 1): m(x)
                ts = []
                for _ in range(reps):
                    if self.dev == "cuda": torch.cuda.synchronize()
                    t0 = time.perf_counter(); m(x)
                    if self.dev == "cuda": torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            fp32 = round(sum(p.numel() for p in m.state_dict().values()) * 4 / 2 ** 20, 1)
            rows.append({"model": n, "params_m": round(self.params[n] / 1e6, 1), "fp32_mb": fp32,
                         "fp16_mb": round(fp32 / 2, 1), "latency_ms_fp32_bs1": round(float(np.median(ts)), 1),
                         "device": torch.cuda.get_device_name(0) if self.dev == "cuda" else "cpu"})
            del m
            if self.dev == "cuda": torch.cuda.empty_cache()
        return pd.DataFrame(rows)
