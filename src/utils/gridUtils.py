from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from collections import Counter

def split(df,target):
    return train_test_split(df,target, test_size=0.1, random_state=42,stratify=target)

def pipeline(model, df, target, SF, numvars, categorical, params):
    est = Counter(target[SF])[0] / Counter(target[SF])[1]
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),
            ('cat', categorical_transformer, categorical)])
    pipe = Pipeline(steps=[('preprocessor', transformations)])
    x_train = pipe.fit_transform(df)
    grid = GridSearchCV(model,
                        params,
                        scoring="f1",
                        cv=5,
                        return_train_score=False,
                        verbose=4,
                        n_jobs=-1)
    grid.fit(x_train, target[SF])
    #class_weights or scale_estimator is gamma='scale in SKLEARN
    return grid, pipe

def evaluateGrid(model, grid, pipe, dfTest,targetTest,SF):
    print(grid.best_estimator_, file=open(f"bestEstimatorParameter_Accuracy{type(model).__name__}_{grid.scoring}.txt", "a"))
    x_test = pipe.transform(dfTest)
    y_test = targetTest[SF]
    y_pred_test = grid.predict(x_test)

    ACC = accuracy_score(
        y_test, y_pred_test
    )

    print("ACCURACY_Test: {}".format(ACC), file=open(f"bestEstimatorParameter_Accuracy{type(model).__name__}_{grid.scoring}.txt", "a"))

    print("CONFUSION MATRIX: {}".format(confusion_matrix(y_test, y_pred_test)),
          file=open(f"bestEstimatorParameter_Accuracy{type(model).__name__}F1.txt", "a"))

    print("ACCURACY_TEST: {}".format(ACC), file=open(f"bestEstimatorParameter_Accuracy{type(model).__name__}_{grid.scoring}.txt", "a"))
    AUC = roc_auc_score(y_test, grid.predict_proba(x_test)[:, 1])
    F1 = f1_score(y_test, grid.predict(x_test))
    Recall = recall_score(y_test, grid.predict(x_test))
    Precision = precision_score(y_test,grid.predict(x_test))
    print("AUC: {}\nF1: {}\nRecall: {}\nPrecision: {}".format(AUC, F1, Recall,Precision),
          file=open(f"bestEstimatorParameter_Accuracy{type(model).__name__}_{grid.scoring}.txt", "a"))
    return 1