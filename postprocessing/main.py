"""
Generate all paper material from experiment logs.

Each figure is saved as a separate PDF for maximum LaTeX flexibility.

Usage:
    python generate_paper_material.py --log_dir ../logs/paper_v1 --out_dir ../figures
    python generate_paper_material.py --log_dir ../logs/paper_v1 --only tables
    python generate_paper_material.py --log_dir ../logs/paper_v1 --only convergence
    python generate_paper_material.py --log_dir ../logs/paper_v1 --only boundaries
    python generate_paper_material.py --log_dir ../logs/paper_v1 --only stopping
    python generate_paper_material.py --log_dir ../logs/paper_v1 --only scaling
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bermudan import *
from bermudan.networks.features import build_features
from bermudan.utils.stopping_times import stopping_times_lsmc, stopping_times_pg

# ─── Default figure style ────────────────────────────────────────────
FIGSIZE = (10, 6)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# ─── Case parameters (must match run_experiments.py) ─────────────────
CASE_A = dict(r=0.06, sigma=0.2, q=0.0, K=40.0, T=1.0, N=50)
CASE_B = dict(r=0.05, q=0.1, sigma=0.2, K=100.0, T=3.0, N=9)
CASE_C = dict(
    r=0.03,
    q=0.0,
    kappa=2.0,
    theta=0.04,
    xi=0.5,
    rho=-0.7,
    K=100.0,
    T=1.0,
    N=48,
    nu_0=0.04,
)


def _build_option(config: dict, cfg: TorchConfig) -> BermudanOption:
    case, d = config["case"], config["d"]
    if case == "A":
        p = CASE_A
        return BermudanOption(
            diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], d=1),
            payoff=Put(K=p["K"]),
            r=p["r"],
            T=p["T"],
            N=p["N"],
            exercise_indices=list(range(1, p["N"] + 1)),
            cfg=cfg,
        )
    elif case.startswith("B") or case == "scaling":
        p = CASE_B
        N = config.get("N_S", p["N"])
        return BermudanOption(
            diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], rho=torch.eye(d), d=d),
            payoff=MaxCall(K=p["K"], d=d),
            r=p["r"],
            T=p["T"],
            N=N,
            exercise_indices=list(range(1, N + 1)),
            cfg=cfg,
        )
    elif case == "C":
        p = CASE_C
        return BermudanOption(
            diffusion=Heston(
                r=p["r"],
                q=p["q"],
                kappa=p["kappa"],
                theta=p["theta"],
                xi=p["xi"],
                rho=p["rho"],
            ),
            payoff=Put(K=p["K"]),
            r=p["r"],
            T=p["T"],
            N=p["N"],
            exercise_indices=list(range(4, p["N"] + 1, 4)),
            cfg=cfg,
        )
    raise ValueError(f"Unknown case: {case}")


def _make_S0(config: dict, cfg: TorchConfig) -> torch.Tensor:
    case, d, s0 = config["case"], config["d"], config["S0"]
    if case == "C":
        return cfg.tensor([float(s0), CASE_C["nu_0"]])
    elif d > 1:
        return cfg.tensor([float(s0)] * d)
    return cfg.tensor([float(s0)])


# ─── Log loading ─────────────────────────────────────────────────────


def load_summary(log_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(log_dir / "summary.csv")
    for col in ["price", "std", "train_time", "eval_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_epochs(log_dir: Path, tag: str) -> pd.DataFrame:
    return pd.read_csv(log_dir / tag / "epochs.csv")


def load_config(log_dir: Path, tag: str) -> dict:
    with open(log_dir / tag / "config.json") as f:
        return json.load(f)


def list_runs(log_dir: Path, method: str = None) -> list[str]:
    tags = []
    for p in sorted(log_dir.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            if method and not p.name.startswith(method):
                continue
            tags.append(p.name)
    return tags


def _load_pg_policy(run_dir: Path, config: dict):
    """Load a PG/A2C policy, inferring architecture from the saved state_dict."""
    # A2C saves as actor.pt, PG saves as policy.pt
    if (run_dir / "actor.pt").exists():
        sd = torch.load(run_dir / "actor.pt", map_location="cpu", weights_only=True)
    elif (run_dir / "policy.pt").exists():
        sd = torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No policy.pt or actor.pt in {run_dir}")

    layer_keys = sorted([k for k in sd if k.endswith(".weight")])
    n_feat = sd[layer_keys[0]].shape[1]
    hidden_dims = [sd[k].shape[0] for k in layer_keys[:-1]]

    from bermudan.networks.feedforward import FeedForward

    net = FeedForward(n_feat, hidden_dims, 1, "relu", None)
    net.load_state_dict(sd)
    net.eval()
    return net


# =====================================================================
# 1. TABLES
# =====================================================================


def generate_tables(df: pd.DataFrame, out: Path):
    print("\n--- TABLES ---")
    out.mkdir(parents=True, exist_ok=True)

    def _fmt(val, std=None):
        if pd.isna(val):
            return "---"
        s = f"{val:.3f}"
        if std is not None and not pd.isna(std) and std > 0:
            s += f" \\small{{({std:.3f})}}"
        return s

    for case in sorted(df["case"].unique()):
        csub = df[df["case"] == case]
        rows = []
        for (d, s0), grp in csub.groupby(["d", "S0"]):
            row = {"$d$": int(d), "$S_0$": int(s0)}
            for _, r in grp.iterrows():
                m = r["method"]
                row[f"{m}"] = _fmt(r["price"], r["std"])
                row[f"{m} time"] = f'{r["train_time"]:.1f}'
            rows.append(row)
        table = pd.DataFrame(rows)
        latex = table.to_latex(
            index=False, escape=False, column_format="c" * len(table.columns)
        )
        path = out / f"table_{case}.tex"
        path.write_text(latex)
        print(f"  {path}")

    # Global summary
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Case": r["case"],
                "$d$": int(r["d"]),
                "$S_0$": int(r["S0"]),
                "Method": r["method"],
                "Price": f'{r["price"]:.3f}',
                "Std": f'{r["std"]:.3f}',
                "Time": f'{r["train_time"]:.1f}',
            }
        )
    path = out / "table_global.tex"
    path.write_text(pd.DataFrame(rows).to_latex(index=False, escape=False))
    print(f"  {path}")


# =====================================================================
# 2. CONVERGENCE (separate figures per metric)
# =====================================================================


def generate_convergence(log_dir: Path, df: pd.DataFrame, out: Path):
    """One PDF per metric per PG run: loss, reward, entropy, grad_norm,
    early_exercise_frac, mean_stop_date."""
    print("\n--- CONVERGENCE ---")
    out.mkdir(parents=True, exist_ok=True)

    refs = {}
    for _, r in df[df["method"] == "LSMC"].iterrows():
        refs[(r["case"], r["d"], r["S0"])] = r["price"]

    for method in ["PG", "A2C"]:
        for tag in list_runs(log_dir, method):
            ep_path = log_dir / tag / "epochs.csv"
            if not ep_path.exists():
                continue
            ep = load_epochs(log_dir, tag)
            cfg = load_config(log_dir, tag)
            ref = refs.get((cfg["case"], cfg["d"], cfg["S0"]))
            label = tag.replace("_", " ")

            # --- Training loss ---
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(ep["epoch"], ep["loss"], lw=0.8, color="C0")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training loss — {label}")
            fig.tight_layout()
            fig.savefig(out / f"loss_{tag}.pdf")
            plt.close(fig)

            # --- Average reward ---
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(
                ep["epoch"], ep["reward_mean"], lw=0.8, color="C0", label="Mean reward"
            )
            if ref is not None:
                ax.axhline(ref, color="red", ls="--", lw=1.0, label=f"LSMC = {ref:.3f}")
                ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reward")
            ax.set_title(f"Average reward — {label}")
            fig.tight_layout()
            fig.savefig(out / f"reward_{tag}.pdf")
            plt.close(fig)

            # --- Mean entropy ---
            if "entropy_mean" in ep.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(ep["epoch"], ep["entropy_mean"], lw=0.8, color="green")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Entropy")
                ax.set_title(f"Mean entropy — {label}")
                fig.tight_layout()
                fig.savefig(out / f"entropy_{tag}.pdf")
                plt.close(fig)

            # --- Gradient norm ---
            if "grad_norm" in ep.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(ep["epoch"], ep["grad_norm"], lw=0.8, color="C3")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gradient norm (before clipping)")
                ax.set_title(f"Gradient norm — {label}")
                fig.tight_layout()
                fig.savefig(out / f"gradnorm_{tag}.pdf")
                plt.close(fig)

            # --- Early exercise fraction ---
            if "early_exercise_frac" in ep.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(ep["epoch"], ep["early_exercise_frac"], lw=0.8, color="C4")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Fraction")
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Early exercise fraction — {label}")
                fig.tight_layout()
                fig.savefig(out / f"early_exercise_{tag}.pdf")
                plt.close(fig)

            # --- Mean stopping date index ---
            if "mean_stop_date" in ep.columns:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(ep["epoch"], ep["mean_stop_date"], lw=0.8, color="C5")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Mean stopping date index")
                ax.set_title(f"Mean stopping date — {label}")
                fig.tight_layout()
                fig.savefig(out / f"mean_stop_{tag}.pdf")
                plt.close(fig)

            print(f"  {tag}: loss, reward, entropy, gradnorm, early_ex, mean_stop")


# =====================================================================
# 3. EXERCISE BOUNDARIES (PG and A2C)
# =====================================================================


def generate_boundaries(log_dir: Path, out: Path):
    print("\n--- BOUNDARIES ---")
    out.mkdir(parents=True, exist_ok=True)

    for method in ["PG", "A2C"]:
        for tag in list_runs(log_dir, method):
            run_dir = log_dir / tag
            has_policy = (run_dir / "policy.pt").exists() or (
                run_dir / "actor.pt"
            ).exists()
            if not has_policy:
                continue
            config = load_config(log_dir, tag)
            if config["case"] == "scaling":
                continue
            d = config["d"]

            if d == 1:
                _plot_boundary_1d(run_dir, config, out, tag)
            elif d == 2:
                _plot_boundary_2d(run_dir, config, out, tag)
            else:
                print(f"  {tag}: d={d}, skip")


def _plot_boundary_1d(run_dir, config, out, tag):
    policy = _load_pg_policy(run_dir, config)
    case = config["case"]
    K = CASE_A["K"] if case == "A" else CASE_C["K"]
    T = CASE_A["T"] if case == "A" else CASE_C["T"]
    N = config.get("N_S", config.get("N", 50))

    s_grid = np.linspace(K * 0.5, K * 1.4, 300)
    t_values = np.linspace(0, T, N + 1)[1:]

    boundary_s, boundary_t = [], []
    with torch.no_grad():
        for t_val in t_values:
            s_t = torch.tensor(s_grid, dtype=torch.float32)
            t_t = torch.full_like(s_t, t_val)
            payoff_val = torch.clamp(K - s_t, min=0.0)
            phi = torch.stack(
                [torch.log(s_t / K + 1e-8), payoff_val / K, t_t / T, (T - t_t) / T],
                dim=-1,
            )
            logit = policy(phi).squeeze(-1).numpy()

            for i in range(len(logit) - 1):
                if logit[i] >= 0 and logit[i + 1] < 0:
                    frac = logit[i] / (logit[i] - logit[i + 1])
                    boundary_s.append(s_grid[i] + frac * (s_grid[i + 1] - s_grid[i]))
                    boundary_t.append(t_val)
                    break

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if boundary_s:
        ax.plot(boundary_t, boundary_s, "b-", lw=1.5, label="PG boundary")
    ax.axhline(K, color="gray", ls=":", lw=1.0, label=f"K = {K:.0f}")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Spot $S$")
    ax.set_title(f"Exercise boundary — {tag.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"boundary_{tag}.pdf")
    plt.close(fig)
    print(f"  {tag}: 1D boundary")


def _plot_boundary_2d(run_dir, config, out, tag):
    policy = _load_pg_policy(run_dir, config)
    d = config["d"]
    K = CASE_B["K"]
    T = CASE_B["T"]
    N_S = config["N_S"]
    t_values = np.linspace(0, T, N_S + 1)[1:]

    if N_S >= 6:
        plot_idx = [0, N_S // 4, N_S // 2, 3 * N_S // 4, N_S - 2, N_S - 1]
    else:
        plot_idx = list(range(min(N_S, 6)))

    s_grid = np.linspace(60, 200, 120)
    S1, S2 = np.meshgrid(s_grid, s_grid)

    # One separate figure per time slice
    for di in plot_idx:
        t_val = t_values[di]
        fig, ax = plt.subplots(figsize=(8, 7))

        with torch.no_grad():
            M = S1.size
            s1 = torch.tensor(S1.flatten(), dtype=torch.float32)
            s2 = torch.tensor(S2.flatten(), dtype=torch.float32)
            S_all = torch.stack([s1, s2], dim=-1)
            eps = 1e-8

            logm = torch.log(S_all / K + eps)  # (M, d)
            max_S = S_all.max(dim=-1).values  # (M,)
            mean_S = S_all.mean(dim=-1)  # (M,)
            top2, _ = torch.topk(S_all, k=2, dim=-1)
            gap = (top2[:, 0] - top2[:, 1]) / K  # (M,)
            t_t = torch.full((M,), t_val, dtype=torch.float32)

            phi = torch.cat(
                [
                    logm,  # (M, d)
                    torch.log(max_S / K + eps).unsqueeze(-1),  # (M, 1)
                    torch.log(mean_S / K + eps).unsqueeze(-1),  # (M, 1)
                    logm.std(dim=-1, keepdim=True),  # (M, 1)
                    gap.unsqueeze(-1),  # (M, 1)
                    (t_t / T).unsqueeze(-1),  # (M, 1)
                    ((T - t_t) / T).unsqueeze(-1),  # (M, 1)
                ],
                dim=-1,
            )  # (M, d+6)

            logit = policy(phi).squeeze(-1).numpy().reshape(S1.shape)
            stop = (logit >= 0).astype(float)

        ax.contourf(S1, S2, stop, levels=[0.5, 1.5], colors=["#FF6B6B"], alpha=0.3)
        ax.contour(S1, S2, stop, levels=[0.5], colors=["red"], linewidths=1.5)
        ax.plot([K, K], [s_grid[0], s_grid[-1]], "k--", lw=0.5)
        ax.plot([s_grid[0], s_grid[-1]], [K, K], "k--", lw=0.5)
        ax.set_xlabel("$S^1$")
        ax.set_ylabel("$S^2$")
        ax.set_title(
            f"Exercise region at $t = {t_val:.2f}$ — " f"{tag.replace('_', ' ')}"
        )
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out / f"boundary_{tag}_t{di}.pdf")
        plt.close(fig)

    print(f"  {tag}: {len(plot_idx)} time slices")


# =====================================================================
# 4. STOPPING TIME DISTRIBUTIONS
# =====================================================================


def generate_stopping_distributions(
    log_dir: Path, df: pd.DataFrame, out: Path, n_paths: int = 200_000
):
    print("\n--- STOPPING DISTRIBUTIONS ---")
    out.mkdir(parents=True, exist_ok=True)

    cfg = TorchConfig.make(device="cpu", dtype=torch.float64, sim_dtype=torch.float32)

    for method in ["PG", "A2C"]:
        for tag in list_runs(log_dir, method):
            run_dir = log_dir / tag
            has_policy = (run_dir / "policy.pt").exists() or (
                run_dir / "actor.pt"
            ).exists()
            if not has_policy:
                continue
            config = load_config(log_dir, tag)
            if config["case"] == "scaling":
                continue

            try:
                option = _build_option(config, cfg)
                S0 = _make_S0(config, cfg)
            except Exception as e:
                print(f"  {tag}: skip ({e})")
                continue

            # Learned policy stopping times
            set_seed(42)
            policy = _load_pg_policy(run_dir, config)
            tau_learned = stopping_times_pg(policy, option, S0, n_paths).cpu().numpy()

            # LSMC stopping times
            set_seed(42)
            tau_lsmc = stopping_times_lsmc(option, S0, n_paths).cpu().numpy()

            # Build bins centered on exercise dates
            ex_dates = option.exercise_dates.cpu().numpy()
            N_S = len(ex_dates)

            def _to_indices(tau, dates):
                idx = np.searchsorted(dates, tau, side="right") - 1
                return np.clip(idx, 0, len(dates) - 1)

            idx_learned = _to_indices(tau_learned, ex_dates)
            idx_lsmc = _to_indices(tau_lsmc, ex_dates)

            bins = np.arange(-0.5, N_S + 0.5, 1)
            method_label = config.get("method", method)

            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.hist(
                idx_lsmc,
                bins=bins,
                alpha=0.5,
                density=True,
                label="LSMC",
                color="C0",
                edgecolor="C0",
                lw=0.5,
            )
            ax.hist(
                idx_learned,
                bins=bins,
                alpha=0.5,
                density=True,
                label=method_label,
                color="C1",
                edgecolor="C1",
                lw=0.5,
            )

            if N_S <= 20:
                tick_idx = list(range(N_S))
            else:
                tick_idx = list(range(0, N_S, max(1, N_S // 10)))
                if N_S - 1 not in tick_idx:
                    tick_idx.append(N_S - 1)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(
                [f"{ex_dates[i]:.2f}" for i in tick_idx], rotation=45, fontsize=9
            )

            ax.set_xlabel("Exercise time $\\tau$")
            ax.set_ylabel("Density")
            ax.set_title(
                f"Stopping distribution — {config['case']}, "
                f"d={config['d']}, $S_0$={config['S0']:.0f}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"stopping_{tag}.pdf")
            plt.close(fig)
            print(f"  {tag}")


# =====================================================================
# 5. SCALING FIGURES
# =====================================================================


def generate_scaling(df: pd.DataFrame, out: Path):
    print("\n--- SCALING ---")
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["case"] == "scaling"]
    if sub.empty:
        print("  No scaling data.")
        return

    for d in sorted(sub["d"].unique()):
        dsub = sub[sub["d"] == d]

        # Time vs N_S
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method, marker, color in [("PG", "o", "C0"), ("DOS", "s", "C1")]:
            m = dsub[dsub["method"] == method].sort_values("N_S")
            if not m.empty:
                ax.plot(
                    m["N_S"],
                    m["train_time"],
                    marker=marker,
                    color=color,
                    lw=1.5,
                    ms=7,
                    label=method,
                )
        ax.set_xlabel("$N_S$ (number of exercise dates)")
        ax.set_ylabel("Training time (s)")
        ax.set_title(f"Computational cost scaling — $d={d}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"scaling_time_d{d}.pdf")
        plt.close(fig)

        # Price vs N_S
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method, marker, color in [("PG", "o", "C0"), ("DOS", "s", "C1")]:
            m = dsub[dsub["method"] == method].sort_values("N_S")
            if not m.empty:
                ax.errorbar(
                    m["N_S"],
                    m["price"],
                    yerr=m["std"],
                    marker=marker,
                    color=color,
                    lw=1.5,
                    ms=7,
                    capsize=3,
                    label=method,
                )
        ax.set_xlabel("$N_S$ (number of exercise dates)")
        ax.set_ylabel("Option price")
        ax.set_title(f"Price convergence — $d={d}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"scaling_price_d{d}.pdf")
        plt.close(fig)

        print(f"  d={d}: time + price")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all paper material")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument(
        "--only",
        default=None,
        choices=["tables", "convergence", "boundaries", "stopping", "scaling"],
    )
    parser.add_argument("--n_paths_stopping", type=int, default=200_000)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    df = load_summary(log_dir)
    print(f"Loaded {len(df)} runs from {log_dir / 'summary.csv'}")

    targets = (
        [args.only]
        if args.only
        else ["tables", "convergence", "boundaries", "stopping", "scaling"]
    )

    if "tables" in targets:
        generate_tables(df, out_dir / "tables")
    if "convergence" in targets:
        generate_convergence(log_dir, df, out_dir / "convergence")
    if "boundaries" in targets:
        generate_boundaries(log_dir, out_dir / "boundaries")
    if "stopping" in targets:
        generate_stopping_distributions(
            log_dir, df, out_dir / "stopping", args.n_paths_stopping
        )
    if "scaling" in targets:
        generate_scaling(df, out_dir / "scaling")

    print(f"\nAll outputs in {out_dir}/")
