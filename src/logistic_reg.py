import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost_reg(X, y, w, b, lambda_=0.1):
    m = X.shape[0]
    h = sigmoid(np.dot(X, w) + b)

    cost = -(1 / m) * np.sum(
        y * np.log(h + 1e-9) +
        (1 - y) * np.log(1 - h + 1e-9)
    )

    reg = (lambda_ / (2 * m)) * np.sum(w ** 2)

    return cost + reg


def compute_gradient_reg(X, y, w, b, lambda_=0.1):
    m = X.shape[0]
    h = sigmoid(np.dot(X, w) + b)

    dj_db = (1 / m) * np.sum(h - y)
    dj_dw = (1 / m) * np.dot(X.T, h - y) + (lambda_ / m) * w

    return dj_db, dj_dw


def train(X_train, y_train, X_val, y_val, alpha=0.1, num_iters=1000, lambda_=0.1):
    w = np.zeros(X_train.shape[1])
    b = 0.0

    train_cost_history = []
    val_cost_history = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_reg(X_train, y_train, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        train_cost = compute_cost_reg(X_train, y_train, w, b, lambda_)
        val_cost = compute_cost_reg(X_val, y_val, w, b, lambda_)

        train_cost_history.append(train_cost)
        val_cost_history.append(val_cost)

        if i % 100 == 0:
            print(
                "Regresion logistica iteracion",
                i,
                "coste_train =",
                round(train_cost, 6),
                "coste_val =",
                round(val_cost, 6)
            )

    return {
        "w": w,
        "b": b,
        "train_cost": train_cost_history,
        "val_cost": val_cost_history
    }


def refit(X_train, y_train, alpha=0.1, num_iters=1000, lambda_=0.1):
    w = np.zeros(X_train.shape[1])
    b = 0.0

    for _i in range(num_iters):
        dj_db, dj_dw = compute_gradient_reg(X_train, y_train, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return {
        "w": w,
        "b": b
    }


def predict(model, X, threshold):
    probabilities = sigmoid(np.dot(X, model["w"]) + model["b"])
    return np.where(probabilities >= threshold, 1, 0)
