import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def save_class_distribution_plot(distributions, path):
    labels = []
    no_values = []
    yes_values = []

    for name, dist in distributions:
        labels.append(name)
        no_values.append(dist["no_rain_percent"])
        yes_values.append(dist["rain_percent"])

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x - 0.2, no_values, width=0.4, label="Clase 0")
    plt.bar(x + 0.2, yes_values, width=0.4, label="Clase 1")
    plt.xticks(x, labels)
    plt.ylabel("Porcentaje")
    plt.title("Distribucion de clases")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def save_cost_plot(train_cost, val_cost, title, path):
    plt.figure()
    plt.plot(train_cost, label="coste train")
    plt.plot(val_cost, label="coste validacion")
    plt.xlabel("Iteracion")
    plt.ylabel("Coste")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def print_tree_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    print()
    print("IMPORTANCIA DE VARIABLES DEL ARBOL")
    print("----------------------------------")
    for i in order[:top_n]:
        print(feature_names[i] + ":", round(float(importances[i]), 6))


def save_pca_iev_cev_analysis(X_train, results_dir, n_components_to_plot=20):
    max_components = min(n_components_to_plot, X_train.shape[1])

    pca = PCA(n_components=max_components)
    pca.fit(X_train)

    iev = pca.explained_variance_ratio_
    cev = np.cumsum(iev)
    x = np.arange(1, len(iev) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, iev, marker="o", label="IEV")
    plt.plot(x, cev, marker="o", label="CEV")
    plt.xlabel("Componente")
    plt.ylabel("Varianza explicada")
    plt.title("PCA: IEV y CEV")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "pca_iev_cev.png")
    plt.close()

