import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    g = sigmoid(z)
    return g * (1 - g)


def initialize_weights(input_size, hidden_size, seed=42):
    rng = np.random.default_rng(seed)

    epsilon1 = np.sqrt(6) / np.sqrt(input_size + hidden_size)
    epsilon2 = np.sqrt(6) / np.sqrt(hidden_size + 1)

    theta1 = rng.uniform(-epsilon1, epsilon1, (hidden_size, input_size + 1))
    theta2 = rng.uniform(-epsilon2, epsilon2, (1, hidden_size + 1))

    return theta1, theta2


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.hstack([np.ones((m, 1)), X])

    z2 = np.dot(a1, theta1.T)
    a2_without_bias = sigmoid(z2)

    a2 = np.hstack([np.ones((m, 1)), a2_without_bias])

    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3).reshape(-1)

    return a1, z2, a2, h


def compute_cost(theta1, theta2, X, y, lambda_=0.01):
    m = X.shape[0]

    _a1, _z2, _a2, h = forward_propagate(X, theta1, theta2)

    cost = -(1 / m) * np.sum(
        y * np.log(h + 1e-9) +
        (1 - y) * np.log(1 - h + 1e-9)
    )

    reg = (lambda_ / (2 * m)) * (
        np.sum(theta1[:, 1:] ** 2) +
        np.sum(theta2[:, 1:] ** 2)
    )

    return cost + reg


def backprop(theta1, theta2, X, y, lambda_=0.01):
    m = X.shape[0]

    a1, z2, a2, h = forward_propagate(X, theta1, theta2)

    d3 = (h - y).reshape(m, 1)
    d2 = np.dot(d3, theta2[:, 1:]) * sigmoid_gradient(z2)

    delta1 = np.dot(d2.T, a1)
    delta2 = np.dot(d3.T, a2)

    grad1 = delta1 / m
    grad2 = delta2 / m

    grad1[:, 1:] = grad1[:, 1:] + (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] = grad2[:, 1:] + (lambda_ / m) * theta2[:, 1:]

    cost = compute_cost(theta1, theta2, X, y, lambda_)

    return cost, grad1, grad2


def train(X_train, y_train, X_val, y_val,
          hidden_size=25, lambda_=0.01, alpha=0.4,
          num_iters=1000, seed=42, patience=150):
    input_size = X_train.shape[1]

    theta1, theta2 = initialize_weights(input_size, hidden_size, seed)

    best_theta1 = theta1.copy()
    best_theta2 = theta2.copy()

    best_val_cost = np.inf
    no_improve = 0

    train_cost_history = []
    val_cost_history = []

    for i in range(num_iters):
        _cost, grad1, grad2 = backprop(theta1, theta2, X_train, y_train, lambda_)

        theta1 = theta1 - alpha * grad1
        theta2 = theta2 - alpha * grad2

        train_cost = compute_cost(theta1, theta2, X_train, y_train, lambda_)
        val_cost = compute_cost(theta1, theta2, X_val, y_val, lambda_)

        train_cost_history.append(train_cost)
        val_cost_history.append(val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            best_theta1 = theta1.copy()
            best_theta2 = theta2.copy()
            no_improve = 0
        else:
            no_improve = no_improve + 1

        if i % 100 == 0:
            print(
                "Red neuronal iteracion",
                i,
                "coste_train =",
                round(train_cost, 6),
                "coste_val =",
                round(val_cost, 6)
            )

        if no_improve >= patience:
            print("Early stopping en iteracion", i)
            break

    return {
        "theta1": best_theta1,
        "theta2": best_theta2,
        "train_cost": train_cost_history,
        "val_cost": val_cost_history
    }


def refit(X_train, y_train,
          hidden_size=25, lambda_=0.01, alpha=0.4,
          num_iters=1000, seed=42):
    input_size = X_train.shape[1]

    theta1, theta2 = initialize_weights(input_size, hidden_size, seed)

    for _i in range(num_iters):
        _cost, grad1, grad2 = backprop(theta1, theta2, X_train, y_train, lambda_)

        theta1 = theta1 - alpha * grad1
        theta2 = theta2 - alpha * grad2

    return {
        "theta1": theta1,
        "theta2": theta2
    }


def predict(model, X, threshold):
    _a1, _z2, _a2, h = forward_propagate(X, model["theta1"], model["theta2"])
    return np.where(h >= threshold, 1, 0)
