from pathlib import Path
import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import preprocessing
import logistic_reg
import red_neuronal
import sklearn_models
import evaluation
import analysis


DATA_PATH = Path("data") / "weatherAUS.csv"
RESULTS_DIR = Path("results")

RANDOM_SEED = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

ALPHA_LOGISTIC = 0.1
LAMBDA_LOGISTIC = 0.1
LOGISTIC_ITERS = 1500

ALPHA_NN = 0.4
LAMBDA_NN = 0.005
NN_ITERS = 1500
NN_HIDDEN_SIZE = 40

KNN_K_VALUES = [3, 5, 7, 9]
TREE_DEPTH_VALUES = [2, 4, 8]
SVM_C_VALUES = [0.1, 1.0, 10.0]

PCA_COMPONENTS_TO_PLOT = 20

SELECTION_METRIC = "f1"
THRESHOLD = 0.46


#Guardar resultados y graficas
def prepare_results_dir():

    #Borrar resultados antiguos
    for path in RESULTS_DIR.iterdir():
        if path.is_file():
            path.unlink()

#Elegir modelo para validacion a raiz de la metrica
def choose_best(validation_rows):
    best_row = validation_rows[0]

    for row in validation_rows[1:]:
        if row[SELECTION_METRIC] > best_row[SELECTION_METRIC]:
            best_row = row

    return best_row


def main():
    prepare_results_dir()

    #Leer dataset
    all_rows = preprocessing.read_weather_csv(DATA_PATH)

    #Balancear dataset
    rows = preprocessing.balanced_sample_rows(
        all_rows,
        n_per_class=15000,
        seed=RANDOM_SEED
    )

    #Dividir train validation test
    train_rows, val_rows, test_rows, train_val_rows = preprocessing.stratified_train_validation_test_split(
        rows,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        seed=RANDOM_SEED,
        max_rows=None
    )
    #Preprocesado
    prep = preprocessing.fit_preprocessor(train_rows)

    #Transformar datos
    X_train, y_train = preprocessing.transform_rows(train_rows, prep)
    X_val, y_val = preprocessing.transform_rows(val_rows, prep)
    X_test, y_test = preprocessing.transform_rows(test_rows, prep)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    if X_train.shape[1] == 0:
        raise ValueError("Error: el dataset se ha quedado con 0 variables de entrada.")

    preprocessing.print_dataset_analysis(rows, y_train, y_val, y_test, prep)

    distributions = [
        ("train", preprocessing.class_distribution(y_train)),
        ("validation", preprocessing.class_distribution(y_val)),
        ("test", preprocessing.class_distribution(y_test))
    ]

    #Generar graficas
    analysis.save_class_distribution_plot(
        distributions,
        RESULTS_DIR / "distribucion_clases.png"
    )

    analysis.save_pca_iev_cev_analysis(
        X_train,
        RESULTS_DIR,
        n_components_to_plot=PCA_COMPONENTS_TO_PLOT
    )

    validation_rows = []

    #Entrenar modelos
    #Regresion logistica
    logistic_model = logistic_reg.train(
        X_train,
        y_train,
        X_val,
        y_val,
        alpha=ALPHA_LOGISTIC,
        num_iters=LOGISTIC_ITERS,
        lambda_=LAMBDA_LOGISTIC
    )

    pred_val = logistic_reg.predict(logistic_model, X_val, threshold=THRESHOLD)
    val_results = evaluation.evaluate_predictions(y_val, pred_val)
    evaluation.print_results("VALIDACION REGRESION LOGISTICA", val_results)

    validation_rows.append(
        evaluation.make_result_row(
            "regresion_logistica",
            "alpha=" + str(ALPHA_LOGISTIC) + ", lambda=" + str(LAMBDA_LOGISTIC),
            val_results
        )
    )

    analysis.save_cost_plot(
        logistic_model["train_cost"],
        logistic_model["val_cost"],
        "Coste regresion logistica",
        RESULTS_DIR / "coste_regresion_logistica.png"
    )

    #Red neuronal
    nn_model = red_neuronal.train(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_size=NN_HIDDEN_SIZE,
        lambda_=LAMBDA_NN,
        alpha=ALPHA_NN,
        num_iters=NN_ITERS,
        seed=RANDOM_SEED
    )

    pred_val = red_neuronal.predict(nn_model, X_val, threshold=THRESHOLD)
    val_results = evaluation.evaluate_predictions(y_val, pred_val)
    evaluation.print_results("VALIDACION RED NEURONAL", val_results)

    validation_rows.append(
        evaluation.make_result_row(
            "red_neuronal",
            "hidden=" + str(NN_HIDDEN_SIZE) +
            ", alpha=" + str(ALPHA_NN) +
            ", lambda=" + str(LAMBDA_NN),
            val_results
        )
    )

    analysis.save_cost_plot(
        nn_model["train_cost"],
        nn_model["val_cost"],
        "Coste red neuronal",
        RESULTS_DIR / "coste_red_neuronal.png"
    )


#ARBOLES DE DECISIÓN
    for depth in TREE_DEPTH_VALUES:
        _tree_model, pred_val = sklearn_models.train_predict_tree(
            X_train,
            y_train,
            X_val,
            max_depth=depth,
            seed=RANDOM_SEED
        )

        val_results = evaluation.evaluate_predictions(y_val, pred_val)
        evaluation.print_results("VALIDACION ARBOL depth=" + str(depth), val_results)

        validation_rows.append(
            evaluation.make_result_row(
                "arbol_decision",
                "max_depth=" + str(depth),
                val_results
            )
        )

# KNN 
    for k in KNN_K_VALUES:
        _model, pred_val = sklearn_models.train_predict_knn(
            X_train,
            y_train,
            X_val,
            k=k
        )

        val_results = evaluation.evaluate_predictions(y_val, pred_val)
        evaluation.print_results("VALIDACION KNN k=" + str(k), val_results)

        validation_rows.append(
            evaluation.make_result_row(
                "knn",
                "k=" + str(k),
                val_results
            )
        )


# SVM
    for c_value in SVM_C_VALUES:
        _model, pred_val = sklearn_models.train_predict_linear_svm(
            X_train,
            y_train,
            X_val,
            c_value=c_value,
            seed=RANDOM_SEED
        )

        val_results = evaluation.evaluate_predictions(y_val, pred_val)
        evaluation.print_results("VALIDACION SVM C=" + str(c_value), val_results)

        validation_rows.append(
            evaluation.make_result_row(
                "svm_lineal",
                "C=" + str(c_value),
                val_results
            )
        )

   #Elegir mejor modelo 
    best = choose_best(validation_rows)

    print()
    print("MEJOR MODELO EN PREVALIDACION")
    print("-----------------------------")
    print("Modelo:", best["model_name"])
    print("Hiperparametros:", best["hyperparameters"])
    print("Criterio de seleccion:", SELECTION_METRIC)
    print("Accuracy validacion:", round(best["accuracy"] * 100, 2), "%")
    print("Balanced accuracy validacion:", round(best["balanced_accuracy"] * 100, 2), "%")
    print("Precision clase 1:", round(best["precision"] * 100, 2), "%")
    print("Recall clase 1:", round(best["recall"] * 100, 2), "%")
    print("F1 clase 1:", round(best["f1"] * 100, 2), "%")
    print("Aciertos validacion:", best["correct"])
    print("Fallos validacion:", best["wrong"])

    #Reentrenar mejor modelo / refit
    final_prep = preprocessing.fit_preprocessor(train_val_rows)
    X_train_val, y_train_val = preprocessing.transform_rows(train_val_rows, final_prep)
    X_test_final, y_test_final = preprocessing.transform_rows(test_rows, final_prep)

    best_model_name = best["model_name"]
    hyperparameters = best["hyperparameters"]

    if best_model_name == "regresion_logistica":
        final_model = logistic_reg.refit(
            X_train_val,
            y_train_val,
            alpha=ALPHA_LOGISTIC,
            num_iters=LOGISTIC_ITERS,
            lambda_=LAMBDA_LOGISTIC
        )
        pred_test = logistic_reg.predict(final_model, X_test_final, threshold=THRESHOLD)

    elif best_model_name == "red_neuronal":
        final_model = red_neuronal.refit(
            X_train_val,
            y_train_val,
            hidden_size=NN_HIDDEN_SIZE,
            lambda_=LAMBDA_NN,
            alpha=ALPHA_NN,
            num_iters=NN_ITERS,
            seed=RANDOM_SEED
        )
        pred_test = red_neuronal.predict(final_model, X_test_final, threshold=THRESHOLD)

    elif best_model_name == "arbol_decision":
        depth = int(hyperparameters.split("=")[1])
        final_model = sklearn_models.refit_tree(
            X_train_val,
            y_train_val,
            max_depth=depth,
            seed=RANDOM_SEED
        )
        pred_test = final_model.predict(X_test_final)

    elif best_model_name == "knn":
        k = int(hyperparameters.split("=")[1])
        final_model = sklearn_models.refit_knn(X_train_val, y_train_val, k=k)
        pred_test = final_model.predict(X_test_final)

    else:
        c_value = float(hyperparameters.split("=")[1])
        final_model = sklearn_models.refit_linear_svm(
            X_train_val,
            y_train_val,
            c_value=c_value,
            seed=RANDOM_SEED
        )
        pred_test = final_model.predict(X_test_final)

    #Evaluar modelo
    test_results = evaluation.evaluate_predictions(y_test_final, pred_test)

    evaluation.print_results("TEST FINAL " + best_model_name, test_results)



if __name__ == "__main__":
    main()
