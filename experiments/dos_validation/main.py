import argparse
import sys
from pathlib import Path
from pprint import pprint

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bermudan import *

# ─────────────────────────────────────────────────────────────────────
# Benchmark configurations from Becker et al. (2019)
# ─────────────────────────────────────────────────────────────────────
COMMON = dict(
    r=0.05,
    q=0.1,  # continuous dividend yield (called delta in the paper)
    sigma=0.2,
    K=100.0,
    T=3.0,
    N=9,  # 9 exercise dates (the simulation grid = exercise grid here)
)

# Reference prices: Table 1 of Becker et al. (2019) / Andersen-Broadie (2004)
BENCHMARKS = {
    # d=2
    (2, 90): 8.075,
    (2, 100): 13.902,
    (2, 110): 21.345,
    # d=5
    (5, 90): 16.644,
    (5, 100): 26.156,
    (5, 110): 36.768,
}


def build_option(d: int, cfg: TorchConfig) -> BermudanOption:
    """Build the Bermudan max-call option for dimension d.

    Becker et al. (2019) use N=9 exercise dates: T/9, 2T/9, ..., T.
    There is NO exercise at t=0.
    """
    N = COMMON["N"]  # 9
    diffusion = GBM(
        r=COMMON["r"],
        sigma=COMMON["sigma"],
        q=COMMON["q"],
        rho=torch.eye(d),
        d=d,
    )
    payoff = MaxCall(K=COMMON["K"], d=d)
    option = BermudanOption(
        diffusion=diffusion,
        payoff=payoff,
        r=COMMON["r"],
        T=COMMON["T"],
        N=N,
        exercise_indices=list(range(1, N + 1)),  # skip t=0
        cfg=cfg,
    )
    return option


def parse_args():
    p = argparse.ArgumentParser(description="Validate DOS on Becker et al. (2019)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--quick", action="store_true", help="Fast smoke test")
    p.add_argument(
        "--dims", nargs="+", type=int, default=[2, 5], help="Dimensions to test"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TorchConfig.make(device=args.device, dtype=torch.float64)

    if args.quick:
        M_lsmc, M_dos = 50_000, 50_000
        dos_iters, dos_bs = 200, 4096
    else:
        M_lsmc, M_dos = 500_000, 500_000
        dos_iters, dos_bs = 1000, 8192

    print(f"Device: {cfg.device}, dtype: {cfg.dtype}")
    print(f"M_lsmc={M_lsmc:,}, M_dos={M_dos:,}, dos_iters={dos_iters}")

    # ─────────────────────────────────────────────────────────────────
    # Part 1: LSMC validation (sanity check)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 1: LSMC validation on Bermudan max-call (Becker et al. 2019)")
    print("=" * 70)
    lsmc = LSMC(degree=2, use_payoff_in_basis=True)

    for d in args.dims:
        option = build_option(d, cfg)
        print(f"\n--- d = {d} ---")
        for s0_val in [90, 100, 110]:
            set_seed(42)
            S0 = cfg.tensor([float(s0_val)] * d)
            res = lsmc.price(option, S0, M_lsmc)
            ref = BENCHMARKS.get((d, s0_val), float("nan"))
            err = abs(res.price - ref) if ref == ref else float("nan")
            print(
                f"  S0={s0_val}:  LSMC={res.price:.3f}  "
                f"ref={ref:.3f}  err={err:.3f}  "
                f"std={res.std:.3f}  time={res.elapsed:.1f}s"
            )

    # ─────────────────────────────────────────────────────────────────
    # Part 2: DOS validation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: DOS validation on Bermudan max-call (Becker et al. 2019)")
    print("=" * 70)
    dos = DOS(
        hidden_dims=None,  # auto: [d+50, d+50]
        lr=1e-3,
        n_iters=dos_iters,
        batch_size=dos_bs,
    )

    for d in args.dims:
        option = build_option(d, cfg)
        print(f"\n--- d = {d} ---")
        for s0_val in [90, 100, 110]:
            set_seed(42)
            S0 = cfg.tensor([float(s0_val)] * d)
            res = dos.price(option, S0, M_dos)
            ref = BENCHMARKS.get((d, s0_val), float("nan"))
            err = abs(res.price - ref) if ref == ref else float("nan")
            print(
                f"  S0={s0_val}:  DOS={res.price:.3f}   "
                f"ref={ref:.3f}  err={err:.3f}  "
                f"std={res.std:.3f}  time={res.elapsed:.1f}s  "
                f"params={res.info['total_params']}"
            )

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REFERENCE PARAMETERS (Becker et al. 2019, Section 4.1)")
    print("=" * 70)
    pprint(COMMON)
    print("Benchmark prices:")
    for (d, s0), ref in sorted(BENCHMARKS.items()):
        print(f"  d={d:3d}, S0={s0:3d} -> {ref:.3f}")
