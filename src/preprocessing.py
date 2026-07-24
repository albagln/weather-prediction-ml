import csv
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "RainTomorrow"
DROP_COLUMNS = ["Date", "RainTomorrow"]


#Guardar filas con RainTomorrow valido
def read_weather_csv(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            target = row.get(TARGET_COLUMN, "")

            if target == "Yes" or target == "No":
                rows.append(row)

    return rows

#Convertir variable obj a 0 o 1
def get_targets_from_rows(rows):
    y = []

    for row in rows:
        if row[TARGET_COLUMN] == "Yes":
            y.append(1)
        else:
            y.append(0)

    return np.array(y)

#Dataset balanceado
def balanced_sample_rows(rows, n_per_class=15000, seed=42):
    y = get_targets_from_rows(rows)
    rng = np.random.default_rng(seed)

    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    selected_class_0 = rng.choice(class_0_idx, size=n_per_class, replace=False)
    selected_class_1 = rng.choice(class_1_idx, size=n_per_class, replace=False)

    selected_idx = np.concatenate([selected_class_0, selected_class_1])
    rng.shuffle(selected_idx)

    balanced_rows = [rows[i] for i in selected_idx]

    return balanced_rows


#Dividir train validation test
def stratified_train_validation_test_split(rows, test_size=0.15, validation_size=0.15,
                                           seed=42, max_rows=30000):
    y = get_targets_from_rows(rows)
    idx = np.arange(len(rows))

    if max_rows is not None and max_rows < len(idx):
        idx, _discard, y, _discard_y = train_test_split(
            idx,
            y,
            train_size=max_rows,
            random_state=seed,
            stratify=y
        )

    selected_rows = [rows[i] for i in idx]
    selected_y = get_targets_from_rows(selected_rows)
    selected_idx = np.arange(len(selected_rows))

    train_val_idx, test_idx, y_train_val, _y_test = train_test_split(
        selected_idx,
        selected_y,
        test_size=test_size,
        random_state=seed,
        stratify=selected_y
    )

    relative_val_size = validation_size / (1.0 - test_size)

    train_idx, val_idx, _y_train, _y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val
    )

    train_rows = [selected_rows[i] for i in train_idx]
    val_rows = [selected_rows[i] for i in val_idx]
    test_rows = [selected_rows[i] for i in test_idx]
    train_val_rows = train_rows + val_rows

    return train_rows, val_rows, test_rows, train_val_rows

#Devolver columnas de entrada
def get_feature_columns(rows):
    all_columns = list(rows[0].keys())
    feature_columns = []

    for col in all_columns:
        if col not in DROP_COLUMNS:
            feature_columns.append(col)

    return feature_columns

#Mirar que columnas son numeros
def is_float_text(value):
    if value is None or value == "" or value == "NA":
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False

#Separar columnas categoricas y numericas
def infer_column_types(train_rows, feature_columns):
    numeric_columns = []
    categorical_columns = []

    for col in feature_columns:
        total = 0
        numeric = 0

        for row in train_rows:
            value = row.get(col, "")

            if value != "" and value != "NA":
                total = total + 1

                if is_float_text(value):
                    numeric = numeric + 1

        if total > 0 and numeric == total:
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    return numeric_columns, categorical_columns

#Ajustar preprocesador con train
def fit_preprocessor(train_rows):
    feature_columns = get_feature_columns(train_rows)
    numeric_columns, categorical_columns = infer_column_types(train_rows, feature_columns)

    means = {}
    sigmas = {}
    categories = {}

    for col in numeric_columns:
        values = []

        for row in train_rows:
            value = row.get(col, "")

            if is_float_text(value):
                values.append(float(value))

        mean = np.mean(values)
        sigma = np.std(values)

        if sigma == 0:
            sigma = 1.0

        means[col] = mean
        sigmas[col] = sigma

    for col in categorical_columns:
        values = []

        for row in train_rows:
            value = row.get(col, "")

            if value != "" and value != "NA":
                values.append(value)

        categories[col] = sorted(set(values))

    feature_names = []

    for col in numeric_columns:
        feature_names.append(col)

    for col in categorical_columns:
        for category in categories[col]:
            feature_names.append(col + "_" + category)

    return {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "means": means,
        "sigmas": sigmas,
        "categories": categories,
        "feature_names": feature_names
    }

#Transformar datos usando preprocesador
def transform_rows(rows, prep):
    X = []
    y = []

    for row in rows:
        features = []

        for col in prep["numeric_columns"]:
            value = row.get(col, "")

            if is_float_text(value):
                number = float(value)
            else:
                number = prep["means"][col]

            normalized = (number - prep["means"][col]) / prep["sigmas"][col]
            features.append(normalized)

        for col in prep["categorical_columns"]:
            value = row.get(col, "")

            for category in prep["categories"][col]:
                if value == category:
                    features.append(1.0)
                else:
                    features.append(0.0)

        X.append(features)

        if row[TARGET_COLUMN] == "Yes":
            y.append(1)
        else:
            y.append(0)

    return np.array(X, dtype=float), np.array(y)

#Cuantos ejemplos hay de cada clase 
def class_distribution(y):
    total = len(y)
    zeros = int(np.sum(y == 0))
    ones = int(np.sum(y == 1))

    return {
        "total": total,
        "no_rain": zeros,
        "rain": ones,
        "no_rain_percent": 100 * zeros / total,
        "rain_percent": 100 * ones / total
    }


#Printear analisis
def print_dataset_analysis(rows, train_y, val_y, test_y, prep):
    print()
    print("ANALISIS DEL DATASET")
    print("--------------------")
    print("Filas con RainTomorrow valido:", len(rows))
    print("Columnas numericas usadas:", len(prep["numeric_columns"]))
    print("Columnas categoricas usadas:", len(prep["categorical_columns"]))
    print("Variables finales despues de one-hot:", len(prep["feature_names"]))


