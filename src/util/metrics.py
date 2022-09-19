import numpy as np
import scipy.stats
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def ci_95(df, exp_metrics, metrics):
    print(metrics)
    print(exp_metrics)
    for experience in df.experience_name.unique():
        df_exp = df[df.experience_name == experience]
        str_experience = " & ".join(experience.split(" - "))
        s = ""
        for col in metrics:
            metrics_values = df_exp[col].values
            val, val_minus, val_plus = mean_confidence_interval(metrics_values)
            # print(col, val, val_minus, val_plus, f"(vs {exp_metrics[col]})")
            s += f" & {round2(val):02} ({round2(val_minus):02}-{round2(val_plus):02})"
        print(str_experience + s + "\\\\")


def round2(n):
    return round(n * 100, 2)


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fn) == 0:
        return -1
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tn + fp) == 0:
        return -1
    return tn / (tn + fp)


def ppv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fp) == 0:
        return -1
    return tp / (tp + fp)


def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tn + fn) == 0:
        return -1
    return tn / (tn + fn)


SCORES = {
    'sensitivity': make_scorer(sensitivity),
    'specificity': make_scorer(specificity),
    'npv': make_scorer(npv),
    'ppv': make_scorer(ppv),
    'accuracy': 'accuracy'
}

SCORES_FUNCT = {
    'sensitivity': sensitivity,
    'specificity': specificity,
    'npv': npv,
    'ppv': ppv,
    'accuracy': accuracy_score
}
