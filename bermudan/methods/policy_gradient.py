import torch
import torch.optim as optim

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult

_LOGIT_CLAMP = 10.0


class PolicyGradient(PricingMethod):
    """REINFORCE with batch-mean baseline for Bermudan options."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        lr: float = 1e-4,
        n_epochs: int = 500,
        batch_size: int = 50_000,
        entropy_coeff: float = 0.05,
        clip_grad_norm: float = 2.0,
    ):
        self.hidden_dims = hidden_dims or [64, 128, 128, 64]
        self.activation = activation
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.clip_grad_norm = clip_grad_norm

    def price(
        self,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
        **kwargs,
    ) -> PricingResult:
        logger = kwargs.get("logger", None)
        cfg = option.cfg
        N_S = option.N_S
        r = option.r
        T = option.T
        payoff = option.payoff
        n_feat = payoff.n_features

        policy = FeedForward(
            input_dim=n_feat,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
        ).to(cfg.device)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr)

        with timer() as train_timer:
            loss_history: list[float] = []
            reward_history: list[float] = []

            for epoch in range(self.n_epochs):
                paths = option.simulate(S0, self.batch_size)
                obs = option.observable_at_exercise(paths)
                ex_dates = option.exercise_dates
                M = self.batch_size

                log_probs_sum = torch.zeros(M, device=cfg.device, dtype=torch.float32)
                entropy_sum = torch.zeros(M, device=cfg.device, dtype=torch.float32)
                stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
                stop_date = torch.full(
                    (M,), N_S - 1, device=cfg.device, dtype=torch.long
                )
                rewards = cfg.zeros(M)

                for n in range(N_S):
                    alive = ~stopped
                    if alive.sum() == 0:
                        break

                    S_n = obs[alive, n, :]
                    t_n = ex_dates[n]
                    phi = build_features(payoff, S_n, t_n, T).float()

                    logit = policy(phi).squeeze(-1)
                    logit = logit.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
                    prob_stop = torch.sigmoid(logit)

                    if n == N_S - 1:
                        action = torch.ones_like(prob_stop)
                    else:
                        action = torch.bernoulli(prob_stop)

                    log_p = action * torch.nn.functional.logsigmoid(logit) + (
                        1 - action
                    ) * torch.nn.functional.logsigmoid(-logit)
                    ent = -(
                        prob_stop * torch.nn.functional.logsigmoid(logit)
                        + (1 - prob_stop) * torch.nn.functional.logsigmoid(-logit)
                    )

                    alive_idx = torch.where(alive)[0]
                    log_probs_sum[alive_idx] += log_p
                    entropy_sum[alive_idx] += ent

                    do_stop = action == 1.0
                    newly_stopped = alive_idx[do_stop]

                    rewards[newly_stopped] = torch.exp(
                        cfg.tensor(-r * t_n.item())
                    ) * payoff(obs[newly_stopped, n, :])
                    stop_date[newly_stopped] = n
                    stopped[newly_stopped] = True

                # Batch mean baseline — exactly as in the working code
                baseline = rewards.mean().detach()
                advantage = rewards - baseline

                pg_loss = -(advantage * log_probs_sum).mean()
                entropy_loss = -self.entropy_coeff * entropy_sum.mean()
                loss = pg_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()

                grad_norm = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm**0.5

                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.clip_grad_norm)
                optimizer.step()

                loss_val = loss.item()
                reward_mean = rewards.mean().item()
                reward_std = rewards.std().item()
                entropy_mean = entropy_sum.mean().item()
                early_ex = (stop_date < N_S - 1).float().mean().item()
                mean_sd = stop_date.float().mean().item()

                loss_history.append(loss_val)
                reward_history.append(reward_mean)

                if logger is not None:
                    logger.log_epoch(
                        epoch=epoch,
                        loss=loss_val,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        entropy_mean=entropy_mean,
                        grad_norm=grad_norm,
                        early_exercise_frac=early_ex,
                        mean_stop_date=mean_sd,
                    )

        with timer() as eval_timer:
            price_val, std_val = self._evaluate(policy, option, S0, n_paths)

        if logger is not None:
            logger.save_model(policy.state_dict(), "policy")

        return PricingResult(
            price=price_val,
            std=std_val,
            elapsed=train_timer.elapsed,
            info={
                "policy": policy,
                "loss_history": loss_history,
                "reward_history": reward_history,
                "n_params": policy.n_params,
                "train_time": train_timer.elapsed,
                "eval_time": eval_timer.elapsed,
            },
        )

    @staticmethod
    @torch.no_grad()
    def _evaluate(
        policy: FeedForward,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
    ) -> tuple[float, float]:
        """Evaluate learned policy greedily on fresh paths."""
        cfg = option.cfg
        policy.eval()

        paths = option.simulate(S0, n_paths)
        obs = option.observable_at_exercise(paths)
        ex_dates = option.exercise_dates
        payoff = option.payoff
        r = option.r
        T = option.T
        M = n_paths
        N_S = option.N_S

        stopped = torch.zeros(M, dtype=torch.bool, device=cfg.device)
        cashflow = cfg.zeros(M)

        for n in range(N_S):
            alive = ~stopped
            if alive.sum() == 0:
                break

            S_n = obs[alive, n, :]
            t_n = ex_dates[n]
            phi = build_features(payoff, S_n, t_n, T).float()
            logit = policy(phi).squeeze(-1)

            if n == N_S - 1:
                do_stop = torch.ones(alive.sum(), dtype=torch.bool, device=cfg.device)
            else:
                do_stop = (logit >= 0) & (payoff(S_n) > 0)

            alive_idx = torch.where(alive)[0]
            stop_idx = alive_idx[do_stop]

            cashflow[stop_idx] = torch.exp(cfg.tensor(-r * t_n.item())) * payoff(
                obs[stop_idx, n, :]
            )
            stopped[stop_idx] = True

        never = ~stopped
        if never.sum() > 0:
            cashflow[never] = torch.exp(cfg.tensor(-r * ex_dates[-1].item())) * payoff(
                obs[never, -1, :]
            )

        policy.train()
        return cashflow.mean().item(), cashflow.std().item() / (M**0.5)
