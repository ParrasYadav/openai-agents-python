from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple


# Simple line world environment with discrete states.
class LineWorld:
    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.position = 0

    def reset(self) -> List[float]:
        self.position = 0
        return [self.position / (self.length - 1)]

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        if action == 1:
            self.position = min(self.position + 1, self.length - 1)
        else:
            self.position = max(self.position - 1, 0)
        reward = 1.0 if self.position == self.length - 1 else -0.1
        done = self.position == self.length - 1
        return [self.position / (self.length - 1)], reward, done


# Simple neural network with one hidden layer implemented using lists.
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.w1 = [
            [random.uniform(-0.1, 0.1) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.b2 = [0.0 for _ in range(output_size)]

    @staticmethod
    def relu(x: float) -> float:
        return x if x > 0.0 else 0.0

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        hidden = [
            self.relu(sum(w_ij * x_j for w_ij, x_j in zip(row, x)) + b)
            for row, b in zip(self.w1, self.b1)
        ]
        output = [
            sum(w_ij * h_j for w_ij, h_j in zip(row, hidden)) + b
            for row, b in zip(self.w2, self.b2)
        ]
        return output, hidden

    def backward(self, x: List[float], hidden: List[float], delta: List[float], lr: float) -> None:
        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                self.w2[i][j] -= lr * delta[i] * hidden[j]
            self.b2[i] -= lr * delta[i]

        dh = [0.0 for _ in range(len(hidden))]
        for j in range(len(hidden)):
            for i in range(len(delta)):
                dh[j] += self.w2[i][j] * delta[i]
            if hidden[j] <= 0:
                dh[j] = 0.0

        for j in range(len(self.w1)):
            for k in range(len(self.w1[j])):
                self.w1[j][k] -= lr * dh[j] * x[k]
            self.b1[j] -= lr * dh[j]


def train(
    env: LineWorld,
    episodes: int,
    gamma: float,
    lr: float,
    epsilon: float,
    hidden_size: int,
) -> float:
    net = NeuralNetwork(1, hidden_size, 2)
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            q_values, hidden = net.forward(state)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = 0 if q_values[0] >= q_values[1] else 1
            next_state, reward, done = env.step(action)
            next_q, _ = net.forward(next_state)
            target = reward + (0.0 if done else gamma * max(next_q))
            delta = [0.0, 0.0]
            delta[action] = q_values[action] - target
            net.backward(state, hidden, delta, lr)
            state = next_state

    total_reward = 0.0
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            q_values, _ = net.forward(state)
            action = 0 if q_values[0] >= q_values[1] else 1
            state, reward, done = env.step(action)
            total_reward += reward
    return total_reward / 10.0


def hyperparameter_search() -> Dict[str, float]:
    env = LineWorld()
    param_grid = {
        "gamma": [0.8, 0.9, 0.99],
        "lr": [0.05, 0.1],
        "epsilon": [0.1, 0.2],
        "hidden_size": [8, 16],
    }
    best_reward = -1e9
    best_params: Dict[str, float] = {}
    for gamma, lr, eps, hidden_size in itertools.product(
        param_grid["gamma"],
        param_grid["lr"],
        param_grid["epsilon"],
        param_grid["hidden_size"],
    ):
        reward = train(env, 100, gamma, lr, eps, hidden_size)
        if reward > best_reward:
            best_reward = reward
            best_params = {
                "gamma": gamma,
                "lr": lr,
                "epsilon": eps,
                "hidden_size": hidden_size,
            }
    return best_params


if __name__ == "__main__":
    result = hyperparameter_search()
    print("Best hyperparameters:", result)

