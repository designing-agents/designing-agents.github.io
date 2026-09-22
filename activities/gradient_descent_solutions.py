"""
Gradient Descent and Backpropagation on MNIST - SOLUTIONS

Reaches 0.953 train and 0.944 test accuracy after 20 epochs, in about 4 seconds
once MNIST is cached.

    pip install datasets numpy pillow matplotlib
    python gradient_descent_solutions.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

N_TRAIN = 60_000
N_TEST = 10_000

SIZES = (784, 16, 16, 10)

LR = 0.1
EPOCHS = 20
BATCH = 64


# --- Task 0 -------------------------------------------------------------------

def load_mnist(n_train: int = N_TRAIN, n_test: int = N_TEST):
    ds = load_dataset("ylecun/mnist")

    def to_arrays(split, n):
        rows = split.select(range(n))
        # Each row's "image" is a PIL image; np.asarray gives a (28, 28) array
        # of uint8, so flatten it and divide by the largest possible pixel.
        X = np.stack([np.asarray(img, dtype=np.float64).ravel()
                      for img in rows["image"]]) / 255.0
        return X, np.asarray(rows["label"], dtype=np.int64)

    X_train, y_train = to_arrays(ds["train"], n_train)
    X_test, y_test = to_arrays(ds["test"], n_test)
    return X_train, y_train, X_test, y_test


# --- Task 1 -------------------------------------------------------------------

@dataclass
class Network:
    Ws: list[np.ndarray]
    bs: list[np.ndarray]


def init_network(rng: np.random.Generator, sizes: tuple = SIZES) -> Network:
    # Scaling by 1/sqrt(fan_in) keeps the pre-activations in the part of the
    # sigmoid that still has slope. Starting at zero instead would make every
    # unit in a layer identical, and identical units receive identical
    # gradients, so they stay clones forever and the layer is worth one unit.
    Ws = [rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
          for fan_in, fan_out in zip(sizes, sizes[1:])]
    bs = [np.zeros(fan_out) for fan_out in sizes[1:]]
    return Network(Ws, bs)


# --- Task 2 -------------------------------------------------------------------

def forward(net: Network, X: np.ndarray) -> list[np.ndarray]:
    A = [X]
    last = len(net.Ws) - 1
    for k, (W, b) in enumerate(zip(net.Ws, net.bs)):
        z = A[-1] @ W + b
        if k == last:
            # Subtracting the row maximum leaves the softmax unchanged, since
            # the constant cancels between numerator and denominator, and it
            # keeps exp() away from overflow.
            e = np.exp(z - z.max(axis=1, keepdims=True))
            A.append(e / e.sum(axis=1, keepdims=True))
        else:
            A.append(1.0 / (1.0 + np.exp(-z)))
    return A


# --- Task 3 -------------------------------------------------------------------

def loss(net: Network, X: np.ndarray, y: np.ndarray) -> float:
    P = forward(net, X)[-1]
    # Pull out the one probability per row that the label points at.
    return float(-np.log(P[np.arange(len(y)), y]).mean())


# --- Task 4 -------------------------------------------------------------------

def numerical_gradient(net: Network, X: np.ndarray, y: np.ndarray,
                       layer: int, i: int, j: int, eps: float = 1e-5) -> float:
    W = net.Ws[layer]
    original = W[i, j]
    W[i, j] = original + eps
    up = loss(net, X, y)
    W[i, j] = original - eps
    down = loss(net, X, y)
    W[i, j] = original          # the caller's network has to come back intact
    # Central difference. The two points are 2 * eps apart, not eps.
    return (up - down) / (2 * eps)


# --- Task 5 -------------------------------------------------------------------

def backward(net: Network, A: list[np.ndarray],
             y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(y)
    Y = np.zeros_like(A[-1])
    Y[np.arange(n), y] = 1.0

    # Softmax and cross-entropy collapse together into this one line. The 1/n
    # is folded in here, which is why the bias gradient below is a plain sum
    # rather than a mean.
    dz = (A[-1] - Y) / n

    dWs: list[np.ndarray] = [np.empty(0)] * len(net.Ws)
    dbs: list[np.ndarray] = [np.empty(0)] * len(net.bs)
    for k in range(len(net.Ws) - 1, -1, -1):
        dWs[k] = A[k].T @ dz
        dbs[k] = dz.sum(axis=0)
        if k > 0:
            # A[k] is layer k's input, which is the sigmoid output of layer
            # k - 1, so its derivative is the one that belongs here. Reaching
            # for A[k + 1] is the usual bug, and Task 4 is what catches it.
            dz = (dz @ net.Ws[k].T) * A[k] * (1.0 - A[k])
    return dWs, dbs


# --- Task 6 -------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray, sizes: tuple = SIZES, lr: float = LR,
          epochs: int = EPOCHS, batch_size: int = BATCH, seed: int = 0,
          net: Network | None = None):
    rng = np.random.default_rng(seed)
    if net is None:
        net = init_network(rng, sizes)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            dWs, dbs = backward(net, forward(net, X[idx]), y[idx])
            # Biases move too, not just weights.
            for k in range(len(net.Ws)):
                net.Ws[k] -= lr * dWs[k]
                net.bs[k] -= lr * dbs[k]
        history.append(loss(net, X, y))
    return net, history


# --- Run (written for you) ----------------------------------------------------
# numpy 2.0 on macOS reports floating point flags that its matrix multiply did
# not actually raise.

warnings.filterwarnings("ignore", message=".*encountered in matmul")


def accuracy(net: Network, X: np.ndarray, y: np.ndarray) -> float:
    """Share of images whose highest-probability digit is the correct one."""
    return float((forward(net, X)[-1].argmax(axis=1) == y).mean())


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    net = init_network(np.random.default_rng(0))

    print(f"{'epoch':>5} {'train loss':>11} {'train acc':>10} "
          f"{'test loss':>10} {'test acc':>9}")
    for epoch in range(1, EPOCHS + 1):
        net, _ = train(X_train, y_train, epochs=1, net=net, seed=epoch)
        print(f"{epoch:>5} "
              f"{loss(net, X_train, y_train):>11.4f} "
              f"{accuracy(net, X_train, y_train):>10.3f} "
              f"{loss(net, X_test, y_test):>10.4f} "
              f"{accuracy(net, X_test, y_test):>9.3f}")


if __name__ == "__main__":
    main()
