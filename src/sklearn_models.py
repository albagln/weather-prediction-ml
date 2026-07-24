from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


#Knn
def train_predict_knn(X_train, y_train, X_eval, k):
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    return model, y_pred

#Arbol de decision
def train_predict_tree(X_train, y_train, X_eval, max_depth, seed=42):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=seed
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    return model, y_pred

#SVM
def train_predict_linear_svm(X_train, y_train, X_eval, c_value, seed=42):
    model = LinearSVC(
        C=c_value,
        random_state=seed,
        max_iter=5000,
        dual=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    return model, y_pred


def refit_knn(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    model.fit(X_train, y_train)
    return model


def refit_tree(X_train, y_train, max_depth, seed=42):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=seed
    )

    model.fit(X_train, y_train)

    return model


def refit_linear_svm(X_train, y_train, c_value, seed=42):
    model = LinearSVC(
        C=c_value,
        random_state=seed,
        max_iter=5000,
        dual=False
    )

    model.fit(X_train, y_train)

    return model
