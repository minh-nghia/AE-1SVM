import numpy as np
from sklearn.metrics import (roc_auc_score, precision_score,
    average_precision_score, recall_score, f1_score, confusion_matrix)


def metrics(y_test, out_y):
    result = dict()
    result['Confusion matrix'] = confusion_matrix(y_test, out_y, labels=[1, -1])
    result['Precision'] = precision_score(y_test, out_y)
    result['Recall'] = recall_score(y_test, out_y)
    result['F1'] = f1_score(y_test, out_y)
    result['AUROC'] = roc_auc_score(y_test, out_y)
    result['AUPRC'] = average_precision_score(y_test, out_y)

    idx = np.argsort(out_y)
    predLabel = np.ones(y_test.shape)

    predLabel[idx[:10]] = -1
    result['P@10'] = precision_score(y_test, predLabel)

    return result