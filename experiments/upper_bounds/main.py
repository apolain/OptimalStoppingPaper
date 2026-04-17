import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bermudan import *
from bermudan.methods.upper_bound import DualBoundResult, andersen_broadie
from bermudan.networks.feedforward import FeedForward

# Reuse config constants from run_experiments
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


def _build_option(config, cfg):
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


def _make_S0(config, cfg):
    case, d, s0 = config["case"], config["d"], config["S0"]
    if case == "C":
        return cfg.tensor([float(s0), CASE_C["nu_0"]])
    elif d > 1:
        return cfg.tensor([float(s0)] * d)
    return cfg.tensor([float(s0)])


def _load_policy(run_dir: Path) -> FeedForward:
    """Load actor/policy from saved state_dict, inferring architecture."""
    for fname in ["actor.pt", "policy.pt"]:
        p = run_dir / fname
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=True)
            keys = sorted([k for k in sd if k.endswith(".weight")])
            n_feat = sd[keys[0]].shape[1]
            hidden = [sd[k].shape[0] for k in keys[:-1]]
            net = FeedForward(n_feat, hidden, 1, "relu", None)
            net.load_state_dict(sd)
            net.eval()
            return net
    raise FileNotFoundError(f"No policy.pt or actor.pt in {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--M_outer", type=int, default=5000)
    parser.add_argument("--M_inner", type=int, default=500)
    parser.add_argument("--methods", nargs="+", default=["PG", "A2C"])
    parser.add_argument("--skip_scaling", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    cfg = TorchConfig.make(
        device=args.device, dtype=torch.float64, sim_dtype=torch.float32
    )

    results = []

    for method in args.methods:
        for run_path in sorted(log_dir.iterdir()):
            if not run_path.is_dir() or not run_path.name.startswith(method):
                continue
            config_path = run_path / "config.json"
            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = json.load(f)

            if args.skip_scaling and config.get("case") == "scaling":
                continue

            tag = run_path.name
            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")

            try:
                policy = _load_policy(run_path).to(cfg.device)
                option = _build_option(config, cfg)
                S0 = _make_S0(config, cfg)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            set_seed(42)
            res = andersen_broadie(
                policy,
                option,
                S0,
                M_outer=args.M_outer,
                M_inner=args.M_inner,
            )

            print(f"  Lower: {res.lower:.4f} ± {res.lower_std:.4f}")
            print(f"  Upper: {res.upper:.4f} ± {res.upper_std:.4f}")
            print(f"  Gap:   {res.gap:.4f}  ({res.gap_relative*100:.2f}%)")
            print(f"  Time:  {res.elapsed:.1f}s")

            results.append(
                {
                    "tag": tag,
                    "method": config["method"],
                    "case": config["case"],
                    "d": config["d"],
                    "S0": config["S0"],
                    "N_S": config.get("N_S", ""),
                    "lower": f"{res.lower:.4f}",
                    "lower_std": f"{res.lower_std:.4f}",
                    "upper": f"{res.upper:.4f}",
                    "upper_std": f"{res.upper_std:.4f}",
                    "gap": f"{res.gap:.4f}",
                    "gap_pct": f"{res.gap_relative*100:.2f}",
                    "time": f"{res.elapsed:.1f}",
                }
            )

    # Save CSV
    import csv

    out_path = log_dir / "upper_bounds.csv"
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{'='*60}")
        print(f"Results saved to {out_path}")
    else:
        print("No runs processed.")


if __name__ == "__main__":
    main()
