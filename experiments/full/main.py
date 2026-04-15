import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bermudan import *

# Case A: 1D Bermudan put under GBM
CASE_A = dict(r=0.06, sigma=0.2, q=0.0, K=40.0, T=1.0, N=50)
CASE_A_SPOTS = [36, 40, 44]
CASE_A_LABELS = {36: "ITM", 40: "ATM", 44: "OTM"}
CASE_A_REFS = {36: 4.486, 40: 2.314, 44: 1.118}

# Case B: max-call GBM symmetric
CASE_B = dict(r=0.05, q=0.1, sigma=0.2, K=100.0, T=3.0, N=9)
CASE_B_SPOTS = [90, 100, 110]
CASE_B_REFS = {
    (2, 90): 8.075,
    (2, 100): 13.902,
    (2, 110): 21.345,
    (5, 90): 16.644,
    (5, 100): 26.156,
    (5, 110): 36.768,
}

# Case C: Heston put
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
CASE_C_SPOTS = [90, 100, 110]
CASE_C_LABELS = {90: "ITM", 100: "ATM", 110: "OTM"}


def build_case_a(cfg):
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


def build_case_b(d, cfg):
    p = CASE_B
    return BermudanOption(
        diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], rho=torch.eye(d), d=d),
        payoff=MaxCall(K=p["K"], d=d),
        r=p["r"],
        T=p["T"],
        N=p["N"],
        exercise_indices=list(range(1, p["N"] + 1)),
        cfg=cfg,
    )


def build_case_b_scaling(d, N_S, cfg):
    """Case B with variable number of exercise dates for scaling experiment."""
    p = CASE_B
    # N = N_S so each simulation step is an exercise date
    return BermudanOption(
        diffusion=GBM(r=p["r"], sigma=p["sigma"], q=p["q"], rho=torch.eye(d), d=d),
        payoff=MaxCall(K=p["K"], d=d),
        r=p["r"],
        T=p["T"],
        N=N_S,
        exercise_indices=list(range(1, N_S + 1)),
        cfg=cfg,
    )


def build_case_c(cfg):
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


def run_method(method, option, S0, n_eval, logger, run_cfg, **price_kwargs):
    """Run a pricing method with full logging."""
    logger.start_run(run_cfg)
    res = method.price(option, S0, n_eval, logger=logger)

    train_time = res.info.get("train_time", res.elapsed)
    eval_time = res.info.get("eval_time", 0.0)
    n_params = res.info.get("n_params", 0)

    logger.end_run(
        price=res.price,
        std=res.std,
        train_time=train_time,
        eval_time=eval_time,
        n_params=n_params,
    )
    return res


def print_result(method_name, res, ref=None):
    err_str = f"  err={abs(res.price - ref):.4f}" if ref else ""
    t_train = res.info.get("train_time", res.elapsed)
    t_eval = res.info.get("eval_time", 0.0)
    n_p = res.info.get("n_params", 0)
    print(
        f"    {method_name:6s}: {res.price:.4f}  std={res.std:.4f}"
        f"{err_str}  train={t_train:.1f}s  eval={t_eval:.1f}s"
        f"  params={n_p}"
    )


def run_case_a(cfg, logger, M_lsmc, M_pg, pg_kwargs, quick=False):
    print("\n" + "=" * 70)
    print("CASE A: 1D Bermudan put under GBM")
    print("=" * 70)
    option = build_case_a(cfg)

    lsmc = LSMC(degree=3, use_payoff_in_basis=False)

    for s0 in CASE_A_SPOTS:
        label = CASE_A_LABELS[s0]
        ref = CASE_A_REFS[s0]
        S0 = cfg.tensor([float(s0)])
        print(f"\n  S0={s0} ({label}), ref={ref:.3f}")

        # LSMC
        set_seed(42)
        rc = RunConfig(
            method="LSMC",
            case="A",
            d=1,
            S0=float(s0),
            N_S=option.N_S,
            N=option.N,
            hyperparams={"degree": 3},
        )
        res = run_method(lsmc, option, S0, M_lsmc, logger, rc)
        print_result("LSMC", res, ref)

        # PG
        set_seed(42)
        kw = dict(pg_kwargs)
        if label == "ITM" and not quick:
            kw["n_epochs"] = max(kw.get("n_epochs", 500), 1000)
            kw["entropy_coeff"] = max(kw.get("entropy_coeff", 0.05), 0.10)
        pg = PolicyGradient(**kw)
        rc = RunConfig(
            method="PG",
            case="A",
            d=1,
            S0=float(s0),
            N_S=option.N_S,
            N=option.N,
            hyperparams=kw,
        )
        res = run_method(pg, option, S0, M_pg, logger, rc)
        print_result("PG", res, ref)


def run_case_b(
    cfg, logger, M_lsmc, M_pg, M_dos, pg_kwargs, dos_kwargs, dims=None, quick=False
):
    print("\n" + "=" * 70)
    print("CASE B: Bermudan max-call under GBM (symmetric)")
    print("=" * 70)
    if dims is None:
        dims = [2, 5]

    lsmc = LSMC(degree=2, use_payoff_in_basis=True)

    for d in dims:
        option = build_case_b(d, cfg)
        print(
            f"\n  --- d={d}, N_S={option.N_S}, features={option.payoff.n_features} ---"
        )

        for s0 in CASE_B_SPOTS:
            ref = CASE_B_REFS.get((d, s0))
            S0 = cfg.tensor([float(s0)] * d)
            ref_str = f"ref={ref:.3f}" if ref else "ref=N/A"
            print(f"\n    S0={s0}, {ref_str}")

            # LSMC
            if d <= 5:
                set_seed(42)
                rc = RunConfig(
                    method="LSMC",
                    case="B_sym",
                    d=d,
                    S0=float(s0),
                    N_S=option.N_S,
                    N=option.N,
                    hyperparams={"degree": 2},
                )
                res = run_method(lsmc, option, S0, M_lsmc, logger, rc)
                print_result("LSMC", res, ref)

            # DOS
            set_seed(42)
            dos = DOS(**dos_kwargs)
            rc = RunConfig(
                method="DOS",
                case="B_sym",
                d=d,
                S0=float(s0),
                N_S=option.N_S,
                N=option.N,
                hyperparams=dos_kwargs,
            )
            res = run_method(dos, option, S0, M_dos, logger, rc)
            print_result("DOS", res, ref)

            # PG
            set_seed(42)
            pg = PolicyGradient(**pg_kwargs)
            rc = RunConfig(
                method="PG",
                case="B_sym",
                d=d,
                S0=float(s0),
                N_S=option.N_S,
                N=option.N,
                hyperparams=pg_kwargs,
            )
            res = run_method(pg, option, S0, M_pg, logger, rc)
            print_result("PG", res, ref)


def run_case_c(cfg, logger, M_lsmc, M_pg, pg_kwargs, quick=False):
    print("\n" + "=" * 70)
    print("CASE C: Bermudan put under Heston")
    print("=" * 70)
    p = CASE_C
    option = build_case_c(cfg)
    print(
        f"  N_S={option.N_S}, N={option.N}, kappa={p['kappa']}, "
        f"theta={p['theta']}, xi={p['xi']}, rho={p['rho']}"
    )

    lsmc = LSMC(degree=3, use_payoff_in_basis=False)

    pg_kw_c = dict(pg_kwargs)
    if not quick:
        pg_kw_c["batch_size"] = max(pg_kw_c.get("batch_size", 200_000), 500_000)

    for s0 in CASE_C_SPOTS:
        label = CASE_C_LABELS[s0]
        S0 = cfg.tensor([float(s0), p["nu_0"]])
        print(f"\n  S0={s0} ({label})")

        # LSMC
        set_seed(42)
        rc = RunConfig(
            method="LSMC",
            case="C",
            d=1,
            S0=float(s0),
            N_S=option.N_S,
            N=option.N,
            hyperparams={"degree": 3},
        )
        res = run_method(lsmc, option, S0, M_lsmc, logger, rc)
        print_result("LSMC", res)

        # PG
        set_seed(42)
        kw = dict(pg_kw_c)
        if label == "ITM" and not quick:
            kw["n_epochs"] = max(kw.get("n_epochs", 500), 1000)
            kw["entropy_coeff"] = max(kw.get("entropy_coeff", 0.05), 0.10)
        pg = PolicyGradient(**kw)
        rc = RunConfig(
            method="PG",
            case="C",
            d=1,
            S0=float(s0),
            N_S=option.N_S,
            N=option.N,
            hyperparams=kw,
        )
        res = run_method(pg, option, S0, M_pg, logger, rc)
        print_result("PG", res)


def run_scaling_ns(cfg, logger, M_pg, M_dos, pg_kwargs, dos_kwargs, quick=False):
    """Scaling experiment: vary N_S at fixed d, measure price + time."""
    print("\n" + "=" * 70)
    print("SCALING: PG vs DOS as a function of N_S")
    print("=" * 70)

    if quick:
        dims = [2]
        ns_values = [9, 20, 50]
    else:
        dims = [2, 10]
        ns_values = [9, 20, 50, 100]

    s0_val = 100  # ATM

    for d in dims:
        print(f"\n  --- d={d}, S0={s0_val} ---")
        print(
            f"  {'N_S':>6s}  {'PG_price':>9s}  {'PG_time':>8s}  "
            f"{'DOS_price':>9s}  {'DOS_time':>8s}"
        )

        for ns in ns_values:
            option = build_case_b_scaling(d, ns, cfg)
            S0 = cfg.tensor([float(s0_val)] * d)

            # Reduce batch for large N_S to avoid OOM
            pg_kw_scaled = dict(pg_kwargs)
            if ns > 50:
                pg_kw_scaled["batch_size"] = min(pg_kw_scaled["batch_size"], 50_000)

            # PG
            set_seed(42)
            pg = PolicyGradient(**pg_kw_scaled)
            rc = RunConfig(
                method="PG",
                case="scaling",
                d=d,
                S0=float(s0_val),
                N_S=ns,
                N=ns,
                hyperparams={**pg_kw_scaled, "experiment": "scaling_ns"},
            )
            res_pg = run_method(pg, option, S0, M_pg, logger, rc)

            # Free memory before DOS
            del pg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # DOS
            set_seed(42)
            dos = DOS(**dos_kwargs)
            rc = RunConfig(
                method="DOS",
                case="scaling",
                d=d,
                S0=float(s0_val),
                N_S=ns,
                N=ns,
                hyperparams={**dos_kwargs, "experiment": "scaling_ns"},
            )
            res_dos = run_method(dos, option, S0, M_dos, logger, rc)

            del dos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pg_t = res_pg.info.get("train_time", res_pg.elapsed)
            dos_t = res_dos.info.get("train_time", res_dos.elapsed)
            print(
                f"  {ns:>6d}  {res_pg.price:>9.3f}  {pg_t:>7.1f}s  "
                f"{res_dos.price:>9.3f}  {dos_t:>7.1f}s"
            )


def parse_args():
    p = argparse.ArgumentParser(description="Run paper experiments")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log_dir", default="logs/experiment")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--cases",
        nargs="+",
        default=["A", "B", "C", "scaling"],
        choices=["A", "B", "C", "scaling"],
    )
    p.add_argument("--b_dims", nargs="+", type=int, default=None)
    p.add_argument("--no_save_models", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TorchConfig.make(
        device=args.device, dtype=torch.float64, sim_dtype=torch.float32
    )
    logger = ExperimentLogger(args.log_dir, save_models=not args.no_save_models)

    if args.quick:
        M_lsmc = 100_000
        M_pg = 50_000
        M_dos = 50_000
        pg_kwargs = dict(
            hidden_dims=[64, 64],
            lr=1e-4,
            n_epochs=100,
            batch_size=50_000,
            entropy_coeff=0.05,
            clip_grad_norm=2.0,
            baseline_ema=0.95,
        )
        dos_kwargs = dict(hidden_dims=None, lr=1e-3, n_iters=200, batch_size=4096)
    else:
        M_lsmc = 200_000
        M_pg = 200_000
        M_dos = 200_000
        pg_kwargs = dict(
            hidden_dims=[256, 256, 256, 128],
            lr=1e-4,
            n_epochs=750,
            batch_size=50_000,
            entropy_coeff=0.05,
            clip_grad_norm=2.0,
        )
        dos_kwargs = dict(hidden_dims=None, lr=1e-3, n_iters=1000, batch_size=8192)

    print(f"Device: {cfg.device}, dtype: {cfg.dtype}, sim_dtype: {cfg.sim_dtype}")
    print(f"Log dir: {args.log_dir}")
    print(f"Cases: {args.cases}")

    if "A" in args.cases:
        run_case_a(cfg, logger, M_lsmc, M_pg, pg_kwargs, quick=args.quick)

    if "B" in args.cases:
        b_dims = args.b_dims or ([2, 5] if not args.quick else [2])
        run_case_b(
            cfg,
            logger,
            M_lsmc,
            M_pg,
            M_dos,
            pg_kwargs,
            dos_kwargs,
            dims=b_dims,
            quick=args.quick,
        )

    if "C" in args.cases:
        run_case_c(cfg, logger, M_lsmc, M_pg, pg_kwargs, quick=args.quick)

    if "scaling" in args.cases:
        run_scaling_ns(
            cfg, logger, M_pg, M_dos, pg_kwargs, dos_kwargs, quick=args.quick
        )

    print("\n" + "=" * 70)
    print(f"ALL DONE — results in {args.log_dir}/summary.csv")
    print("=" * 70)
