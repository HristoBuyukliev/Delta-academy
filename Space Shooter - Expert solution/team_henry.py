from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical
from collections import deque

from tqdm import tqdm

from delta_shooter.game_mechanics import ShooterEnv, choose_move_randomly, save_network, load_network, play_shooter, \
    human_player

TEAM_NAME = "Henry"


def get_gamlam_matrix(gamma: float, lamda: float, batch_size: int):
    gamlam = gamma * lamda

    gamlam_geo_series = torch.tensor([gamlam ** n for n in range(batch_size)])

    # Shift the coefficients to the right for each successive row
    full_gamlam_matrix = torch.stack([torch.roll(gamlam_geo_series, shifts=n) for n in range(batch_size)])

    # Sets everything except upper-triangular to 0
    return torch.triu(full_gamlam_matrix)


def calculate_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        successor_values: torch.Tensor,
        is_terminals: torch.Tensor,
        gamma: float,
        full_gamlam_matrix: torch.Tensor,
):
    """
    Calculate the Generalized Advantage Estimator (GAE) for a batch of transitions.

    GAE = \sum_{t=0}^{T-1} (gamma * lamda)^t * (r_{t+1} + gamma * V_{t+1} - V_t)
    """
    # Gets the delta terms: the TD-errors
    delta_terms = rewards + gamma * successor_values - values

    # Zero out terms that are after an episode termination
    for terminal_index in torch.squeeze(is_terminals.nonzero(), dim=1):
        full_gamlam_matrix[: terminal_index + 1, terminal_index + 1:] = 0

    return torch.matmul(full_gamlam_matrix, delta_terms)


def train(pol_net: Optional[nn.Module] = None, val_net: Optional[nn.Module] = None):
    # Hyperparameters
    gamma = 0.99
    lamda = 0.95
    epsilon = 0.05
    entropy_coeff = 0.025

    batch_size = 1024  # Bigger batch sizes are more stable, but take longer to train
    num_episodes = 50_000
    epochs_per_batch = 3

    pol_lr = 0.002
    n_pol_neurons = 128
    val_lr = 0.01
    n_val_neurons = 128

    N_INPUTS = 24
    N_ACTIONS = 6

    # Calculate gamlam matrix - this saves computation when calculating GAE
    gamlam_matrix = get_gamlam_matrix(gamma, lamda, batch_size)

    # Policy net
    policy = pol_net or nn.Sequential(
        nn.Linear(N_INPUTS, n_pol_neurons),
        nn.LeakyReLU(),  # LeakyReLU stops the "dying ReLU" problem
        nn.Linear(n_pol_neurons, n_pol_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons, N_ACTIONS),
        nn.Softmax(dim=-1),
    )

    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)

    # Value net
    V = val_net or nn.Sequential(
        nn.Linear(N_INPUTS, n_val_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons, n_val_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons, n_val_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons, 1),
    )

    val_optim = torch.optim.Adam(V.parameters(), lr=val_lr)
    val_loss_fn = nn.MSELoss()

    eps_per_debug = 100
    game = ShooterEnv(choose_move_randomly, include_barriers=True, half_sized_game=False)

    past_step_counts = deque(maxlen=1000)
    past_winrates = deque(maxlen=1000)
    batch = []

    for ep_num in tqdm(range(num_episodes)):

        num_steps = 0
        state, reward, done, _ = game.reset()

        while not done:
            prev_state = state
            action_probs = Categorical(policy(state))
            action = action_probs.sample()
            state, reward, done, _ = game.step(action.item())
            batch.append((prev_state, action, reward, state, done))
            num_steps += 1

            if len(batch) >= batch_size:
                states = torch.stack([item[0] for item in batch])
                actions = torch.tensor([item[1] for item in batch], requires_grad=False)
                rewards = torch.tensor([item[2] for item in batch], requires_grad=False)
                successor_states = torch.stack([item[3] for item in batch])
                is_terminals = torch.tensor([item[4] for item in batch], requires_grad=False)

                # Value update
                for _ in range(epochs_per_batch):
                    vals = torch.squeeze(V(states))
                    det_vals = vals.clone().detach()
                    with torch.no_grad():
                        successor_vals = (
                                torch.squeeze(V(successor_states)) * ~is_terminals
                        )

                    gae = calculate_gae(rewards, det_vals, successor_vals, is_terminals, gamma, gamlam_matrix.clone())
                    lambda_returns = gae + det_vals

                    val_loss = val_loss_fn(vals, lambda_returns)

                    val_optim.zero_grad()
                    val_loss.backward()
                    val_optim.step()

                # Policy update
                if ep_num > 1000:
                    with torch.no_grad():
                        old_pol_probs = policy(states)[range(batch_size), actions]

                        vals = torch.squeeze(V(states))
                        successor_vals = (
                                torch.squeeze(V(successor_states)) * ~is_terminals
                        )
                        gae = calculate_gae(rewards, vals, successor_vals, is_terminals, gamma, gamlam_matrix.clone())

                    for _ in range(epochs_per_batch):
                        pol_dist = Categorical(policy(states))
                        pol_probs = pol_dist.probs[range(batch_size), actions]
                        clipped_obj = torch.clip(pol_probs / old_pol_probs, 1 - epsilon, 1 + epsilon)

                        ppo_obj = (
                                torch.min(clipped_obj * gae, (pol_probs / old_pol_probs) * gae)
                                + entropy_coeff * pol_dist.entropy()
                        )
                        pol_loss = -torch.sum(ppo_obj)

                        pol_optim.zero_grad()
                        pol_loss.backward()
                        pol_optim.step()

                batch = []

        past_winrates.append(reward == 1)
        past_step_counts.append(num_steps)
        if ep_num % eps_per_debug == eps_per_debug - 1:
            print(
                "Ep:",
                ep_num + 1,
                "Avg steps:",
                round(sum(past_step_counts) / len(past_step_counts), 2),
                "Winrate:",
                round(sum(past_winrates) / len(past_winrates), 2),
                )

        # Stopping condition
        if len(past_winrates) > 1000 and sum(past_winrates) / len(past_winrates) >= 0.98:
            save_network(policy, f"{TEAM_NAME}_stopping_condition")
            save_network(V, f"{TEAM_NAME}_stopping_condition_value")
            break

    return policy, V


def evaluate_policy(policy: nn.Module, half_sized_game: bool = True, include_barriers: bool = False, n_games: int = 20):
    n_wins, n_losses = 0, 0
    for _ in range(n_games):
        total_return = play_shooter(
            your_choose_move=lambda x: choose_move(x, policy),
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=1,
            render=True,
            include_barriers=include_barriers,
            half_game_size=half_sized_game,
        )
        if total_return > 0:
            print("You won!")
            n_wins += 1
        else:
            print("You lost!")
            n_losses += 1
    print(f"Win rate: {n_wins / (n_wins + n_losses)}")


def choose_move(
    state: torch.Tensor,
    neural_network: nn.Module,
) -> int:
    return torch.argmax(neural_network(state)).item()


if __name__ == "__main__":
    # Example workflow, feel free to edit this! ###
    # my_network, value_net = train(load_network(f"{TEAM_NAME}"), load_network(f"{TEAM_NAME}"))
    # save_network(my_network, TEAM_NAME)
    # Save value function so can be used in training if we restart it
    # save_network(value_net, f"{TEAM_NAME}_value")

    my_network = load_network(f"{TEAM_NAME}")

    def choose_move_no_network(state) -> int:
        """The arguments in play_pong's game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    # evaluate_policy(my_network, half_sized_game=False, include_barriers=True, n_games=20)

    # The code below plays a single game against your bot.
    # You play as the pink ship
    play_shooter(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        include_barriers=True,
        half_game_size=False,
    )
