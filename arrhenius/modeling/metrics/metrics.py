import torch
import torch.nn as nn
from enum import Enum
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score


class Dir(str, Enum):
    FOR = "for"
    REV = "rev"


class Tgt(str, Enum):
    A10 = "A10"
    N = "n"
    EAY = "EaY"
    LNK = "lnk"
    A = "A"
    EA = "Ea"


def _key(x):
    # Works for Enum or plain str
    return x.value if hasattr(x, "value") else str(x)


class MetricRegistry(nn.Module):
    _triplet_cols = {Tgt.A10: 0, Tgt.N: 1, Tgt.EAY: 2}

    def __init__(
        self,
        targets_cols,
        metrics_builders,
        splits=("val", "test"),
        include_lnk: bool = False,
        namespace: str | None = None,  # <- NEW: lets us avoid key collisions
    ):
        super().__init__()
        self.targets_cols = dict(targets_cols)  # e.g., {Tgt.A10:0, Tgt.N:1, Tgt.EAY:2}
        if include_lnk:
            self.targets_cols[Tgt.LNK] = None  # special: no column, flat input

        self.splits = splits
        self.metrics_builders = metrics_builders
        self.namespace = namespace or ""
        self.store = nn.ModuleDict()
        for sp in splits:  # e.g., "val", "test"
            d_sp = nn.ModuleDict()
            for d in Dir:  # Dir.FOR, Dir.REV
                d_dir = nn.ModuleDict()
                for tgt in self.targets_cols.keys():
                    d_dir[_key(tgt)] = nn.ModuleDict(
                        {name: cls() for name, cls in metrics_builders.items()}
                    )
                d_sp[_key(d)] = d_dir
            self.store[sp] = d_sp

    def update_scaled_triplet(self, split, direction, preds_s, targets_s):
        # alias with shape checks
        if preds_s.ndim != 2 or preds_s.size(1) != 3:
            raise ValueError(f"update_scaled_triplet expects (B,3), got {tuple(preds_s.shape)}")
        return self.update_scaled(split, direction, preds_s, targets_s)

    def update_batch(
        self,
        split: str,
        *,
        scaled: dict[str, tuple[torch.Tensor, torch.Tensor]],
        lnk: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        """
        scaled: {"for": (Y_pred_s, Y_true_s), "rev": (Y_pred_s, Y_true_s)} ; each (B,3)
        lnk:    {"for": (lnK_pred, lnK_true), "rev": (lnK_pred, lnK_true)} ; any shape → flattened
        """
        Yf_p, Yf_t = scaled["for"]
        Yr_p, Yr_t = scaled["rev"]

        self.update_scaled_triplet(split, Dir.FOR, Yf_p, Yf_t)
        self.update_scaled_triplet(split, Dir.REV, Yr_p, Yr_t)

        if lnk is not None and (Tgt.LNK in self.targets_cols):
            lf_p, lf_t = lnk["for"]
            lr_p, lr_t = lnk["rev"]
            self.update_lnk(split, Dir.FOR, lf_p, lf_t)
            self.update_lnk(split, Dir.REV, lr_p, lr_t)

    def update_scaled(self, split, direction, preds_s, targets_s):
        """preds_s/targets_s: (B,3) scaled triplet"""
        for tgt, col in self.targets_cols.items():
            if tgt == Tgt.LNK:  # skip here, handled separately
                continue
            self._update_one(split, direction, tgt, preds_s[:, col], targets_s[:, col])

    def update_lnk(self, split, direction, preds, targets):
        if Tgt.LNK in self.targets_cols:
            self._update_one(split, direction, Tgt.LNK, preds.flatten(), targets.flatten())

    def _update_one(self, split, direction: Dir, tgt: Tgt, preds, targets):
        bucket = self.store[split][_key(direction)][_key(tgt)]
        for stat, metric in bucket.items():
            metric.update(preds, targets)

    def log_and_reset(self, pl_module, split):
        # base metric namespace: e.g., "val_raw" or just "val"
        base = f"{split}{('_' + self.namespace) if self.namespace else ''}"
        for tgt in self.targets_cols.keys():
            key_t = _key(tgt)
            stats_for = {
                stat: m.compute() for stat, m in self.store[split][_key(Dir.FOR)][key_t].items()
            }
            stats_rev = {
                stat: m.compute() for stat, m in self.store[split][_key(Dir.REV)][key_t].items()
            }
            for stat in stats_for.keys():
                pl_module.log(f"{base}/{stat}_{key_t}_for", stats_for[stat], on_epoch=True)
                pl_module.log(f"{base}/{stat}_{key_t}_rev", stats_rev[stat], on_epoch=True)
                pl_module.log(
                    f"{base}/{stat}_{key_t}_avg",
                    0.5 * (stats_for[stat] + stats_rev[stat]),
                    on_epoch=True,
                )

        # reset
        for d in Dir:
            for tgt in self.targets_cols.keys():
                for m in self.store[split][_key(d)][_key(tgt)].values():
                    m.reset()
