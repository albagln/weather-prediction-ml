# 🌦️ Weather Prediction with Machine Learning

Predicting next-day rainfall in Australia using machine learning techniques and the WeatherAUS dataset.

---

# 🇬🇧 English

## Overview

This project explores binary classification techniques to predict whether it will rain the next day (`RainTomorrow`) using the **WeatherAUS** meteorological dataset.

The project includes custom implementations of machine learning algorithms, data preprocessing, model evaluation and comparisons with equivalent implementations from **scikit-learn**.

* **Class 0:** `RainTomorrow = 0`
* **Class 1:** `RainTomorrow = 1`

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Dataset

This project uses the publicly available **WeatherAUS** dataset from Kaggle.

The dataset is **not included** in this repository due to its size.

Download it from Kaggle and place it in:

```text
data/weatherAUS.csv
```

---

## Running the Project

After downloading the dataset:

```text
python main.py
```

---

## Project Structure

```text
main.py
preprocessing.py
logistic_reg.py
red_neuronal.py
sklearn_models.py
evaluation.py
analysis.py
```

---

## Evaluation Metrics

The project reports the following metrics in the terminal:

```text
Accuracy
Balanced Accuracy
Precision (Class 1)
Recall (Class 1)
F1-score (Class 1)
Correct Predictions
Incorrect Predictions
```

Although Accuracy and the number of correct/incorrect predictions are easy to interpret, the dataset is imbalanced, with significantly more samples belonging to `RainTomorrow = 0`.

For this reason, the final model is selected using **Balanced Accuracy**, preventing the majority class from dominating the evaluation.

---

## Generated Results

The following figures are automatically saved inside the `results/` directory:

```text
coste_regresion_logistica.png
coste_red_neuronal.png
pca_iev_cev.png
distribucion_clases.png
```

All remaining evaluation metrics are displayed in the terminal.


---

## Future Improvements

Possible future improvements include:

* Hyperparameter optimization
* Cross-validation
* Additional machine learning models
* Model persistence
* Docker support
* Automated testing

---

# 🇪🇸 Español

# 🌦️ Predicción de lluvia mediante Machine Learning

Predicción de lluvia para el día siguiente utilizando técnicas de aprendizaje automático y el conjunto de datos **WeatherAUS**.

---

# 🇪🇸 Español

## Descripción

Este proyecto explora diferentes técnicas de clasificación binaria para predecir si lloverá al día siguiente (`RainTomorrow`) utilizando el conjunto de datos meteorológico **WeatherAUS**.

El proyecto incluye implementaciones propias de algoritmos de Machine Learning, el preprocesado del conjunto de datos, la evaluación de los modelos y la comparación de su rendimiento con implementaciones equivalentes de **scikit-learn**.

* **Clase 0:** `RainTomorrow = 0`
* **Clase 1:** `RainTomorrow = 1`

---

## Tecnologías

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Dataset

Este proyecto utiliza el conjunto de datos público **WeatherAUS**, disponible en Kaggle.

Debido a su tamaño, el dataset **no está incluido** en este repositorio.

Puedes descargarlo desde Kaggle y colocarlo en:

```text
data/weatherAUS.csv
```

---

## Ejecución del proyecto

Una vez descargado el dataset, ejecuta:

```text
python main.py
```

---

## Estructura del proyecto

```text
main.py
preprocessing.py
logistic_reg.py
red_neuronal.py
sklearn_models.py
evaluation.py
analysis.py
```

---

## Métricas de evaluación

El proyecto muestra por terminal las siguientes métricas:

```text
Accuracy
Balanced Accuracy
Precision (Clase 1)
Recall (Clase 1)
F1-score (Clase 1)
Aciertos
Fallos
```

Aunque la **Accuracy** y el número de aciertos y fallos son métricas fáciles de interpretar, el conjunto de datos está desbalanceado, ya que contiene muchos más ejemplos pertenecientes a `RainTomorrow = 0` que a `RainTomorrow = 1`.

Por este motivo, el modelo final se selecciona utilizando **Balanced Accuracy**, evitando que la clase mayoritaria condicione completamente la evaluación del rendimiento.

---

## Resultados generados

Las siguientes gráficas se guardan automáticamente en la carpeta `results/`:

```text
coste_regresion_logistica.png
coste_red_neuronal.png
pca_iev_cev.png
distribucion_clases.png
```

El resto de métricas y resultados se muestran por terminal.

---

## Mejoras futuras

Algunas mejoras que podrían incorporarse en el futuro son:

* Optimización de hiperparámetros.
* Validación cruzada (*Cross-Validation*).
* Incorporación de nuevos modelos de Machine Learning.
* Persistencia de modelos entrenados.
* Contenerización mediante Docker.
* Automatización de pruebas.

