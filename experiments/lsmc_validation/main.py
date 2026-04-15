import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bermudan import *

# Case A: 1D Bermudan put under GBM
CASE_A_PARAMS = dict(r=0.06, sigma=0.2, q=0.0, K=40.0, T=1.0, N=50)
CASE_A_CONFIGS = [
    {"S0": 36, "label": "ITM", "ref": 4.486},
    {"S0": 40, "label": "ATM", "ref": 2.314},
    {"S0": 44, "label": "OTM", "ref": 1.118},
]


def build_case_a(cfg: TorchConfig) -> BermudanOption:
    p = CASE_A_PARAMS
    return BermudanOption(
        diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], d=1),
        payoff=Put(K=p["K"]),
        r=p["r"],
        T=p["T"],
        N=p["N"],
        exercise_indices=list(range(1, p["N"] + 1)),
        cfg=cfg,
    )


# Case B: Bermudan max-call under GBM
CASE_B_PARAMS = dict(r=0.05, q=0.1, sigma=0.2, K=100.0, T=3.0, N=9)
CASE_B_REFS = {
    (2, 90): 8.075,
    (2, 100): 13.902,
    (2, 110): 21.345,
    (5, 90): 16.644,
    (5, 100): 26.156,
    (5, 110): 36.768,
}


def build_case_b(d: int, cfg: TorchConfig) -> BermudanOption:
    p = CASE_B_PARAMS
    return BermudanOption(
        diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], rho=torch.eye(d), d=d),
        payoff=MaxCall(K=p["K"], d=d),
        r=p["r"],
        T=p["T"],
        N=p["N"],
        exercise_indices=list(range(1, p["N"] + 1)),
        cfg=cfg,
    )


# Case C: Bermudan put under Heston
CASE_C_PARAMS = dict(
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
CASE_C_CONFIGS = [
    {"S0": 90, "label": "ITM"},
    {"S0": 100, "label": "ATM"},
    {"S0": 110, "label": "OTM"},
]


def build_case_c(cfg: TorchConfig) -> BermudanOption:
    p = CASE_C_PARAMS
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


# Runners
def run_case_a(cfg, M, degree):
    print("\n" + "=" * 70)
    print("CASE A: 1D Bermudan put under GBM")
    print("=" * 70)
    p = CASE_A_PARAMS
    option = build_case_a(cfg)
    print(f"  N_S={option.N_S}, T={p['T']}, K={p['K']}, sigma={p['sigma']}")

    lsmc = LSMC(degree=degree, use_payoff_in_basis=False)

    for c in CASE_A_CONFIGS:
        set_seed(42)
        S0 = cfg.tensor([float(c["S0"])])
        res = lsmc.price(option, S0, M)
        err = abs(res.price - c["ref"])
        print(f"\n  S0={c['S0']} ({c['label']}):  ref={c['ref']:.3f}")
        print(
            f"    LSMC: {res.price:.4f}  std={res.std:.4f}  "
            f"err={err:.4f}  time={res.elapsed:.1f}s"
        )


def run_case_b(cfg, M, degree, dims):
    print("\n" + "=" * 70)
    print("CASE B: Bermudan max-call under GBM")
    print("=" * 70)

    lsmc = LSMC(degree=degree, use_payoff_in_basis=True)

    for d in dims:
        option = build_case_b(d, cfg)
        print(f"\n  --- d={d}, N_S={option.N_S} ---")

        for s0_val in [90, 100, 110]:
            set_seed(42)
            ref = CASE_B_REFS.get((d, s0_val))
            S0 = cfg.tensor([float(s0_val)] * d)
            res = lsmc.price(option, S0, M)

            ref_str = f"ref={ref:.3f}" if ref else "ref=N/A"
            err_str = f"err={abs(res.price - ref):.4f}" if ref else ""
            print(f"\n    S0={s0_val}: {ref_str}")
            print(
                f"      LSMC: {res.price:.4f}  std={res.std:.4f}  "
                f"{err_str}  time={res.elapsed:.1f}s"
            )


def run_case_c(cfg, M, degree):
    print("\n" + "=" * 70)
    print("CASE C: Bermudan put under Heston")
    print("=" * 70)
    p = CASE_C_PARAMS
    option = build_case_c(cfg)
    print(
        f"  N_S={option.N_S}, N={option.N}, kappa={p['kappa']}, "
        f"theta={p['theta']}, xi={p['xi']}, rho={p['rho']}"
    )

    lsmc = LSMC(degree=degree, use_payoff_in_basis=False)

    for c in CASE_C_CONFIGS:
        set_seed(42)
        S0 = cfg.tensor([float(c["S0"]), p["nu_0"]])
        res = lsmc.price(option, S0, M)
        print(f"\n  S0={c['S0']} ({c['label']}):")
        print(f"    LSMC: {res.price:.4f}  std={res.std:.4f}  time={res.elapsed:.1f}s")


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Validate LSMC implementation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--cases", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"]
    )
    p.add_argument("--b_dims", nargs="+", type=int, default=None)
    p.add_argument("--degree", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TorchConfig.make(
        device=args.device, dtype=torch.float64, sim_dtype=torch.float32
    )

    M = 100_000 if args.quick else 500_000

    print(f"Device: {cfg.device}, dtype: {cfg.dtype}, sim_dtype: {cfg.sim_dtype}")
    print(f"M={M:,}, degree={args.degree}")

    if "A" in args.cases:
        run_case_a(cfg, M, args.degree)

    if "B" in args.cases:
        b_dims = args.b_dims or ([2] if args.quick else [2, 5])
        run_case_b(cfg, M, args.degree, b_dims)

    if "C" in args.cases:
        run_case_c(cfg, M, args.degree)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
