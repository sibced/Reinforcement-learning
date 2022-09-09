import copy
from itertools import count

import numpy as np
import torch
from pybulletgym.envs.roboschool.envs.pendulum.inverted_double_pendulum_env import (
    InvertedDoublePendulumBulletEnv,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from utils import (
    compute_rewards_to_go,
    estimate_advantage,
    expected_cumulative_return,
    sample_trajectory,
)


def do_our_ppo(
    policy_network,
    value_network,
    prefix,
    *,
    n_actors: int,
    max_len: int,
    n_epochs: int,
    batch_size: int,
    n_iteration: int,
    eps: float,
):

    writer = SummaryWriter()

    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-5)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=1e-5)

    counter = count()

    # with gym.make("InvertedPendulum-v2") as env:
    with InvertedDoublePendulumBulletEnv() as env:

        for iteration in range(n_iteration):
            print(f"{iteration=}/{n_iteration=}")

            policy_network.eval()
            value_network.eval()

            model_old = copy.deepcopy(policy_network)
            model_old.eval()

            # collect N trajectories
            with torch.no_grad():
                (full_observations, actions, rewards) = zip(
                    *(
                        sample_trajectory(
                            env, model_old.get_policy(epsilon=0.3), max_len
                        )
                        for _ in range(n_actors)
                    )
                )

                # rewards_torch = list(map(torch.from_numpy, rewards))
                full_observations_torch = list(map(torch.from_numpy, full_observations))
                actions_torch = list(map(torch.from_numpy, actions))

                # build training set
                rewards_to_go = [
                    compute_rewards_to_go(rewards_) for rewards_ in rewards
                ]

            writer.add_scalar(
                prefix + "Max length", max(len(a) for a in actions), iteration
            )
            # writer.add_scalar(
            #     prefix + "Max RTG0", max(rtg[0] for rtg in rewards_to_go), iteration
            # )
            # if hasattr(policy_network, "log_std"):
            #     std = torch.exp(policy_network.log_std)
            #     writer.add_scalar(prefix + "Module.STD", std, iteration)

            # train value network
            policy_network.eval()
            value_network.train()

            # build dataset
            observations_torch = torch.cat(
                [
                    torch.from_numpy(observations_[:-1])
                    for observations_ in full_observations
                ]
            )
            actions_torch = torch.cat(actions_torch)
            rewards_to_go = torch.from_numpy(np.concatenate(rewards_to_go))

            dataset = TensorDataset(
                observations_torch,
                rewards_to_go,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            counter_bkp = copy.deepcopy(counter)

            for epoch in range(n_epochs):

                glob_step = next(counter)
                writer.add_scalar(prefix + "epoch", epoch, glob_step)

                loss_value_lst = torch.zeros((len(dataloader),), dtype=torch.float32)

                for batch_idx, (o, rtg) in enumerate(dataloader):

                    value_optimizer.zero_grad()

                    loss_value = (value_network(o) - rtg) ** 2
                    loss_value_scalar = loss_value.mean()

                    loss_value_lst[batch_idx] = loss_value_scalar.detach()

                    loss_value_scalar.backward()

                    value_optimizer.step()

                writer.add_scalar(prefix + "ValueLoss", loss_value_scalar, glob_step)

            value_network.eval()
            policy_network.eval()

            # collection N trajectories part 2
            with torch.no_grad():
                full_observation_values = [
                    value_network(obs) for obs in full_observations_torch
                ]

                advantage = [
                    estimate_advantage(val_, reward_, lambda_=0.95)
                    for val_, reward_ in zip(full_observation_values, rewards)
                ]

            # train
            policy_network.train()
            value_network.eval()

            # build a random batch of data
            advantage = torch.from_numpy(np.concatenate(advantage))

            dataset = TensorDataset(
                observations_torch,
                actions_torch,
                advantage,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # restore (so the policy updates are synced in tensorboard)
            counter = counter_bkp

            for epoch in range(n_epochs):

                glob_step = next(counter)

                loss_clip_lst = torch.zeros((len(dataloader),), dtype=torch.float32)
                ratio_min_lst = torch.zeros((len(dataloader),), dtype=torch.float32)
                ratio_max_lst = torch.zeros((len(dataloader),), dtype=torch.float32)

                for batch_idx, (o, u, adv) in enumerate(dataloader):

                    policy_optimizer.zero_grad()

                    with torch.no_grad():
                        prob_old = model_old.posterior(o, u)
                    prob = policy_network.posterior(o, u)

                    ratio = prob / prob_old

                    ratio_min_lst[batch_idx] = ratio.min().detach()
                    ratio_max_lst[batch_idx] = ratio.max().detach()

                    TRPO_loss = ratio * adv

                    clipped_adv = torch.where(adv < 0, (1 - eps) * adv, (1 + eps) * adv)

                    loss_clip = torch.minimum(TRPO_loss, clipped_adv)
                    # pytorch does gradient *descent* -> flip sign
                    loss_clip_scalar = -loss_clip.mean()

                    loss_clip_lst[batch_idx] = loss_clip_scalar.detach()

                    loss_clip_scalar.backward()

                    policy_optimizer.step()

                writer.add_scalar(  # log the value that should increase
                    prefix + "PolicyLoss", -loss_clip_scalar, glob_step
                )

                writer.add_scalar(prefix + "RatioMin", ratio_min_lst.mean(), glob_step)
                writer.add_scalar(prefix + "RatioMax", ratio_max_lst.mean(), glob_step)

                policy_network.eval()

                with torch.no_grad():
                    ecr = expected_cumulative_return(
                        env, policy_network.get_policy(epsilon=0), 200, 50
                    )
                    writer.add_scalar(prefix + "ECR", ecr, glob_step)
