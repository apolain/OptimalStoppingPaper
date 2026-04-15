import torch
import torch.nn.functional as F
import torch.optim as optim

from ..networks.features import build_features
from ..networks.feedforward import FeedForward
from ..options.bermudan import BermudanOption
from ..utils.timing import timer
from .base import PricingMethod, PricingResult

_LOGIT_CLAMP = 10.0


class ActorCritic(PricingMethod):
    """A2C with state-dependent baseline for Bermudan options.

    Parameters
    ----------
    actor_dims : list[int]
        Hidden layer widths of the actor (policy) network.
    critic_dims : list[int]
        Hidden layer widths of the critic (value) network.
    activation : str
        Activation for hidden layers.
    lr_actor : float
        Learning rate for the actor.
    lr_critic : float
        Learning rate for the critic.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Paths per epoch.
    entropy_coeff : float
        Entropy regularisation coefficient.
    clip_grad_norm : float
        Max gradient norm for both actor and critic.
    critic_coeff : float
        Weight of the critic loss relative to the actor loss.
    """

    def __init__(
        self,
        actor_dims: list[int] | None = None,
        critic_dims: list[int] | None = None,
        activation: str = "relu",
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 500,
        batch_size: int = 50_000,
        entropy_coeff: float = 0.05,
        clip_grad_norm: float = 2.0,
        critic_coeff: float = 0.5,
    ):
        self.actor_dims = actor_dims or [256, 256, 256, 128]
        self.critic_dims = critic_dims or [128, 128]
        self.activation = activation
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.clip_grad_norm = clip_grad_norm
        self.critic_coeff = critic_coeff

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

        # Actor: logit output
        actor = FeedForward(
            input_dim=n_feat,
            hidden_dims=self.actor_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
        ).to(cfg.device)

        # Critic: scalar value output
        critic = FeedForward(
            input_dim=n_feat,
            hidden_dims=self.critic_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
        ).to(cfg.device)

        optimizer_actor = optim.Adam(actor.parameters(), lr=self.lr_actor)
        optimizer_critic = optim.Adam(critic.parameters(), lr=self.lr_critic)

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

                # Collect value estimates at the first decision point for each path
                value_estimates = torch.zeros(M, device=cfg.device, dtype=torch.float32)
                first_features = None

                for n in range(N_S):
                    alive = ~stopped
                    if alive.sum() == 0:
                        break

                    S_n = obs[alive, n, :]
                    t_n = ex_dates[n]
                    phi = build_features(payoff, S_n, t_n, T).float()

                    # Store features at each path's first alive date for critic
                    if n == 0:
                        first_features = phi.clone()
                        v_pred = critic(phi).squeeze(-1)
                        value_estimates[:] = v_pred

                    # Actor forward
                    logit = actor(phi).squeeze(-1)
                    logit = logit.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
                    prob_stop = torch.sigmoid(logit)

                    if n == N_S - 1:
                        action = torch.ones_like(prob_stop)
                    else:
                        action = torch.bernoulli(prob_stop)

                    log_p = action * F.logsigmoid(logit) + (1 - action) * F.logsigmoid(
                        -logit
                    )
                    ent = -(
                        prob_stop * F.logsigmoid(logit)
                        + (1 - prob_stop) * F.logsigmoid(-logit)
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

                # --- Advantage using critic baseline ---
                advantage = rewards.float() - value_estimates.detach()

                # Actor loss
                actor_loss = -(advantage.detach() * log_probs_sum).mean()
                entropy_loss = -self.entropy_coeff * entropy_sum.mean()

                # Critic loss (MSE between predicted value and realised reward)
                critic_loss = F.mse_loss(value_estimates, rewards.float().detach())

                # Combined update
                total_loss = actor_loss + entropy_loss + self.critic_coeff * critic_loss

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                total_loss.backward()

                grad_norm_actor = 0.0
                for p in actor.parameters():
                    if p.grad is not None:
                        grad_norm_actor += p.grad.data.norm(2).item() ** 2
                grad_norm_actor = grad_norm_actor**0.5

                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.clip_grad_norm)

                optimizer_actor.step()
                optimizer_critic.step()

                # Metrics
                loss_val = total_loss.item()
                reward_mean = rewards.mean().item()
                reward_std = rewards.std().item()
                entropy_mean = entropy_sum.mean().item()
                early_ex = (stop_date < N_S - 1).float().mean().item()
                mean_sd = stop_date.float().mean().item()
                critic_loss_val = critic_loss.item()

                loss_history.append(loss_val)
                reward_history.append(reward_mean)

                if logger is not None:
                    logger.log_epoch(
                        epoch=epoch,
                        loss=loss_val,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        entropy_mean=entropy_mean,
                        grad_norm=grad_norm_actor,
                        early_exercise_frac=early_ex,
                        mean_stop_date=mean_sd,
                        critic_loss=critic_loss_val,
                    )

        with timer() as eval_timer:
            price_val, std_val = self._evaluate(actor, option, S0, n_paths)

        n_params_actor = actor.n_params
        n_params_critic = critic.n_params

        if logger is not None:
            logger.save_model(actor.state_dict(), "actor")
            logger.save_model(critic.state_dict(), "critic")

        return PricingResult(
            price=price_val,
            std=std_val,
            elapsed=train_timer.elapsed,
            info={
                "actor": actor,
                "critic": critic,
                "loss_history": loss_history,
                "reward_history": reward_history,
                "n_params": n_params_actor,
                "n_params_critic": n_params_critic,
                "train_time": train_timer.elapsed,
                "eval_time": eval_timer.elapsed,
            },
        )

    @staticmethod
    @torch.no_grad()
    def _evaluate(
        actor: FeedForward,
        option: BermudanOption,
        S0: torch.Tensor,
        n_paths: int,
    ) -> tuple[float, float]:
        """Evaluate learned policy greedily on fresh paths."""
        cfg = option.cfg
        actor.eval()

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
            logit = actor(phi).squeeze(-1)

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

        actor.train()
        return cashflow.mean().item(), cashflow.std().item() / (M**0.5)
