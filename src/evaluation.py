import numpy as np

#dividir sin error por 0
def divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator

#Calcula metricas del modelo
def evaluate_predictions(y_true, y_pred):
    total = len(y_true)

    correct = int(np.sum(y_true == y_pred))
    wrong = int(np.sum(y_true != y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = divide(correct, total)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "total": int(total),
        "correct": correct,
        "wrong": wrong,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

#Imprimir en terminal
def print_results(name, results):
    print()
    print("=====", name, "=====")
    print("Total ejemplos:", results["total"])
    print("Aciertos:", results["correct"])
    print("Fallos:", results["wrong"])
    print("Accuracy:", round(results["accuracy"] * 100, 2), "%")
    print("Balanced accuracy:", round(results["balanced_accuracy"] * 100, 2), "%")
    print("Precision clase 1:", round(results["precision"] * 100, 2), "%")
    print("Recall clase 1:", round(results["recall"] * 100, 2), "%")
    print("F1 clase 1:", round(results["f1"] * 100, 2), "%")

#Fila resumen para mejor modelo
def make_result_row(model_name, hyperparameters, results):
    return {
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "total": results["total"],
        "correct": results["correct"],
        "wrong": results["wrong"],
        "accuracy": results["accuracy"],
        "balanced_accuracy": results["balanced_accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"]
    }
