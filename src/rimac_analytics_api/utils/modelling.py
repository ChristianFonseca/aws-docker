import pandas as pd
import numpy as np
from sklearn.metrics import (recall_score, accuracy_score, precision_score, auc, roc_curve)
import matplotlib.pyplot as plt


def filter_threshold(probabilities, threshold):
    """
    Target predicted based on the threshold.

    Parameters
    ----------
    probabilities: list, series or iterable object.
        Probabilities of a model.
    threshold: numeric.
        Value for split target predicted.

    Returns
    -------
    list.
    """
    return [1 if f >= threshold else 0 for f in probabilities]


def get_threshold_measures_df(observed, probabilities, steps=0.05):
    """
    Get metrics for observed variables and its estimated probabilities.

    Parameters
    ----------
    observed: list, series or iterable object.
        Real target (1 or 0).
    probabilities: list, series or iterable object.
        Probabilities of a model.
    steps: numeric (between 0 and 1), list.
        If numeric between 0 and 1: step value to evaluate matrics from threshold 0 to 1.
        If list: list of values to evaluate matrics.

    Returns
    -------
    Dataframe.
    """
    # steps *= 100
    # steps = [x / 100.0 for x in range(steps, 100, steps)]
    if isinstance(steps, float):
        steps = np.arange(0, 1, steps)
    df = pd.DataFrame(columns=['Punto de corte', 'N_Predicted', 'Recall', 'Accuracy', 'Precision'])

    for i in range(len(steps)):
        estimated_threshold = filter_threshold(probabilities, steps[i])
        row = [
            steps[i],
            sum(estimated_threshold),
            recall_score(observed, estimated_threshold),
            accuracy_score(observed, estimated_threshold),
            precision_score(observed, estimated_threshold),
        ]
        df.loc[i] = row

    return df


def obtener_AUC_Gini_Roc(target, predict):
    """
    Plot roc curve.

    Parameters
    ----------
    target: Target variable.
    predict: Predicted variable.

    Returns
    -------
    Plot
    """
    fpr, tpr, _ = roc_curve(target, predict)
    roc_auc = auc(fpr, tpr)
    gini = 2*roc_auc - 1
    print('AUC: {}'.format(roc_auc))
    print('GINI: {}'.format(gini))
    plt.rcParams['figure.figsize'] = (10, 10)
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def feature_importance(importance, features, plot_n_first=False):
    """
    Return / Plot feature importance.

    Parameters
    ----------
    importance: List, array.
        Importances of features
    features: List, array.
        Features of the model
    plot_n_first: bool, int.
        If int, plot the top features based on the importance.

    Returns
    -------
    Plot
    """
    importance = {features[a]: importance[a] for a in range(0, len(features))}
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    if plot_n_first:
        noms = pd.DataFrame(importance)[0]
        vals = pd.DataFrame(importance)[1]
        n = int(plot_n_first)
        fig, ax = plt.subplots(figsize=(18, 9))
        x = noms[:n]  # list(reversed(cols[:n]))
        y = vals[:n]  # list(reversed(vals[:n]))
        ax.barh(np.arange(len(x)), y)
        ax.set_yticks(np.arange(len(x)))
        ax.set_yticklabels(x)
        ax.invert_yaxis()
        ax.set_title('Importancia de la variable')
        ax.set_xlabel('Importancia')
        plt.show()

    return importance
