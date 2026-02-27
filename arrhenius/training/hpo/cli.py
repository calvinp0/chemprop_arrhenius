# run_hpo/cli.py

import argparse


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf-path", required=True)
    p.add_argument("--target-csv", required=True)
    p.add_argument("--rad-dir")
    p.add_argument(
        "--rad-source",
        choices=["path", "rad", "default"],
        default="path",
        help="RAD/geometry CSV schema to load when extra_mode uses atom extras.",
    )
    p.add_argument("--search-space", default="search_space.yaml")
    p.add_argument(
        "--extra-mode",
        choices=[
            "baseline",
            "geom_only",
            "local",
            "atom",
            "rad",
            "rad_local",
            "rad_local_noc",
            "none",
            "path",
            "default",
        ],
        default="baseline",
    )
    p.add_argument(
        "--global-mode",
        choices=["none", "morgan_count", "morgan_binary", "rdkit2d_norm", "rdkit2d", "auto"],
        default="none",
    )
    p.add_argument("--morgan-bits", type=int, default=2048)
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--study-name", default="arrhenius_hpo")
    p.add_argument("--storage-url", default="sqlite:///arrhenius_hpo.db")
    p.add_argument("--hpo-trials", type=int, default=0)
    p.add_argument("--temp-min", type=float, default=300.0)
    p.add_argument("--temp-max", type=float, default=3100.0)
    p.add_argument("--temp-step", type=float, default=100.0)
    p.add_argument(
        "--metrics-db",
        default="metrics.sqlite",
        help="SQLite file to store trial metrics/config/scaler stats/split indices.",
    )
    p.add_argument(
        "--save-raw-scalers",
        action="store_true",
        help="Also persist raw scaler vectors (may be large).",
    )
    p.add_argument(
        "--arr-layer",
        choices=["on", "off"],
        default="on",
        help="Enable or disable Arrhenius layer in the model (default: on).",
    )
    p.add_argument(
        "--arr-supervision",
        choices=["on", "off"],
        default="on",
        help="Include ln k(T) supervision loss when Arrhenius layer is enabled (default: on).",
    )
    p.add_argument("--seed", type=int, default=42, help="Global seed for full reproducibility")
    p.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip launching new HPO trials and reuse existing study results.",
    )
    p.add_argument(
        "--top-k-final",
        type=int,
        default=3,
        help="Number of best configs to evaluate with the post-HPO 10-fold sweep.",
    )
    p.add_argument(
        "--locked-test-frac",
        type=float,
        default=0.0,
        help="If > 0, reserve this fraction as a strict locked test set before HPO/CV.",
    )
    p.add_argument(
        "--rank-metric",
        default=None,
        help="Validation metric used to rank top-k after 10-fold evaluation "
        "(default resolves to search_space.rank_metric_default, else val/mae_lnk_avg).",
    )
    p.add_argument(
        "--relax-topk",
        action="store_true",
        help="If fewer than top-k trials match the current split signature, "
        "fall back to best overall completed trials.",
    )
    p.add_argument(
        "--final-test-frac",
        type=float,
        default=0.15,
        help="Fraction of data reserved for the final hold-out test set.",
    )
    p.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Skip post-HPO top-k evaluation and hold-out retraining.",
    )
    p.add_argument(
        "--final-metrics-db",
        default="final_metrics.sqlite",
        help="SQLite file where the reproducibility-focused evaluations are stored.",
    )
    p.add_argument(
        "--final-study-name",
        default="arrhenius_final",
        help="Study name recorded inside the final metrics database.",
    )
    p.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Hardware accelerator passed to Lightning (auto detects GPU when available).",
    )
    p.add_argument(
        "--devices",
        default="1",
        help="`devices` argument for Lightning Trainer (e.g. 1, 2, or 'auto').",
    )
    p.add_argument(
        "--precision",
        default=None,
        help="Optional Lightning precision override (e.g. '16-mixed', '32-true').",
    )
    p.add_argument(
        "--splitter",
        choices=["kstone", "random"],
        default="kstone",
        help="Select the splitter used for inner/hold-out splits (Kennard-Stone or grouped random).",
    )
    p.add_argument(
        "--outer-folds",
        type=int,
        default=3,
        help="Number of outer folds built for cross-validation during HPO.",
    )
    p.add_argument(
        "--reset-split-exports",
        action="store_true",
        help="Remove any existing split export CSV/SQLite before writing new results.",
    )
    p.add_argument(
        "--record-final-preds",
        action="store_true",
        default=True,
        help="Record per-sample predictions (scaled and unscaled) during the final top-k CV sweep.",
    )
    p.add_argument(
        "--no-record-final-preds",
        action="store_false",
        dest="record_final_preds",
        help="Disable recording per-sample predictions during the final top-k CV sweep.",
    )
    return p
