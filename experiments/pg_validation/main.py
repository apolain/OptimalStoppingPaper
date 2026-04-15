import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bermudan import *

# Case A: 1D Bermudan put under GBM (Longstaff-Schwartz 2001)
CASE_A_PARAMS = dict(r=0.06, sigma=0.2, q=0.0, K=40.0, T=1.0, N=50)
CASE_A_S0 = {
    36: {"label": "ITM", "ref": 4.486},
    40: {"label": "ATM", "ref": 2.314},
    44: {"label": "OTM", "ref": 1.118},
}


def build_case_a(cfg: TorchConfig) -> BermudanOption:
    p = CASE_A_PARAMS
    return BermudanOption(
        diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], d=1),
        payoff=Put(K=p["K"]),
        r=p["r"],
        T=p["T"],
        N=p["N"],
        exercise_indices=list(range(1, p["N"] + 1)),  # skip t=0
        cfg=cfg,
    )


# Case B: Bermudan max-call under GBM (Becker et al. 2019)
CASE_B_PARAMS = dict(r=0.05, q=0.1, sigma=0.2, K=100.0, T=3.0, N=9)
CASE_B_REFS = {
    # (d, S0) -> reference price
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
CASE_C_S0 = {90: "ITM", 100: "ATM", 110: "OTM"}


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
        # 12 exercise dates evenly spaced (monthly), skip t=0
        exercise_indices=list(range(4, p["N"] + 1, 4)),
        cfg=cfg,
    )


# Runner
def run_case_a(cfg, M_lsmc, M_pg, pg_kwargs, quick=False):
    print("\n" + "=" * 70)
    print("CASE A: 1D Bermudan put under GBM")
    print("=" * 70)
    option = build_case_a(cfg)
    print(f"  N_S={option.N_S} exercise dates, T={option.T}, K={CASE_A_PARAMS['K']}")

    lsmc = LSMC(degree=3, use_payoff_in_basis=False)

    # PG hyperparameters
    pg_base = dict(pg_kwargs)

    for s0_val, info in CASE_A_S0.items():
        label, ref = info["label"], info["ref"]
        S0 = cfg.tensor([float(s0_val)])

        # LSMC
        set_seed(42)
        res_l = lsmc.price(option, S0, M_lsmc)

        # PG
        set_seed(42)
        kw = dict(pg_base)
        if label == "ITM" and not quick:
            kw["n_epochs"] = kw.get("n_epochs", 300) * 2
            kw["entropy_coeff"] = 0.02
        pg = PolicyGradient(**kw)
        res_p = pg.price(option, S0, M_pg)

        print(f"\n  S0={s0_val} ({label}):  ref={ref:.3f}")
        print(
            f"    LSMC: {res_l.price:.3f}  (err={abs(res_l.price-ref):.3f}, {res_l.elapsed:.1f}s)"
        )
        print(
            f"    PG:   {res_p.price:.3f}  (err={abs(res_p.price-ref):.3f}, {res_p.elapsed:.1f}s, "
            f"params={res_p.info['n_params']})"
        )


def run_case_b(cfg, M_lsmc, M_pg, pg_kwargs, dims=None, quick=False):
    print("\n" + "=" * 70)
    print("CASE B: Bermudan max-call under GBM")
    print("=" * 70)

    if dims is None:
        dims = [2, 5]

    lsmc = LSMC(degree=2, use_payoff_in_basis=True)

    for d in dims:
        option = build_case_b(d, cfg)
        print(
            f"\n  --- d = {d}, N_S={option.N_S}, features={option.payoff.n_features} ---"
        )

        for s0_val in [90, 100, 110]:
            ref = CASE_B_REFS.get((d, s0_val), float("nan"))
            S0 = cfg.tensor([float(s0_val)] * d)

            # LSMC
            if d <= 5:
                set_seed(42)
                res_l = lsmc.price(option, S0, M_lsmc)
                lsmc_str = f"LSMC: {res_l.price:.3f} ({res_l.elapsed:.1f}s)"
            else:
                lsmc_str = "LSMC: skipped (d too high)"

            # PG
            set_seed(42)
            pg = PolicyGradient(**pg_kwargs)
            res_p = pg.price(option, S0, M_pg)

            ref_str = f"ref={ref:.3f}" if ref == ref else "ref=N/A"
            err_str = f"err={abs(res_p.price-ref):.3f}" if ref == ref else ""
            print(f"\n    S0={s0_val}: {ref_str}")
            print(f"      {lsmc_str}")
            print(
                f"      PG:   {res_p.price:.3f}  ({err_str}, {res_p.elapsed:.1f}s, "
                f"params={res_p.info['n_params']})"
            )


def run_case_c(cfg, M_lsmc, M_pg, pg_kwargs, quick=False):
    print("\n" + "=" * 70)
    print("CASE C: Bermudan put under Heston")
    print("=" * 70)
    p = CASE_C_PARAMS
    option = build_case_c(cfg)
    print(f"  N_S={option.N_S} exercise dates, N={option.N} sim steps, T={p['T']}")
    print(f"  kappa={p['kappa']}, theta={p['theta']}, xi={p['xi']}, rho_SV={p['rho']}")

    lsmc = LSMC(degree=3, use_payoff_in_basis=False)

    pg_base = dict(pg_kwargs)
    if not quick:
        pg_base["batch_size"] = max(pg_base.get("batch_size", 200_000), 500_000)

    for s0_val, label in CASE_C_S0.items():
        S0 = cfg.tensor([float(s0_val), p["nu_0"]])

        set_seed(42)
        res_l = lsmc.price(option, S0, M_lsmc)

        set_seed(42)
        kw = dict(pg_base)
        if label == "ITM" and not quick:
            kw["n_epochs"] = kw.get("n_epochs", 300) * 2
            kw["lr"] = 1e-4
        pg = PolicyGradient(**kw)
        res_p = pg.price(option, S0, M_pg)

        print(f"\n  S0={s0_val} ({label}):")
        print(f"    LSMC: {res_l.price:.3f}  ({res_l.elapsed:.1f}s)")
        print(
            f"    PG:   {res_p.price:.3f}  ({res_p.elapsed:.1f}s, "
            f"params={res_p.info['n_params']})"
        )


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Validate Policy Gradient")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true", help="Fast smoke test")
    p.add_argument(
        "--cases", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"]
    )
    p.add_argument(
        "--b_dims",
        nargs="+",
        type=int,
        default=None,
        help="Dimensions for Case B (default: 2 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TorchConfig.make(device=args.device, dtype=torch.float64)

    if args.quick:
        M_lsmc = 100_000
        M_pg = 50_000
        pg_kwargs = dict(
            hidden_dims=[32, 32],
            lr=1e-3,
            n_epochs=50,
            batch_size=50_000,
            entropy_coeff=0.01,
            clip_grad_norm=1.0,
        )
    else:
        M_lsmc = 500_000
        M_pg = 500_000
        pg_kwargs = dict(
            hidden_dims=[64, 64],
            lr=1e-3,
            n_epochs=300,
            batch_size=200_000,
            entropy_coeff=0.01,
            clip_grad_norm=1.0,
        )

    print(f"Device: {cfg.device}, dtype: {cfg.dtype}")
    print(f"PG config: {pg_kwargs}")

    if "A" in args.cases:
        run_case_a(cfg, M_lsmc, M_pg, pg_kwargs, quick=args.quick)

    if "B" in args.cases:
        b_dims = args.b_dims or ([2] if args.quick else [2, 5])
        run_case_b(cfg, M_lsmc, M_pg, pg_kwargs, dims=b_dims, quick=args.quick)

    if "C" in args.cases:
        run_case_c(cfg, M_lsmc, M_pg, pg_kwargs, quick=args.quick)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
