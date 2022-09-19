from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedGroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from CONSTANTS import NARIN_RESULTS, NARIN_FIGURES, LABEL_FILE
from CONSTANTS import TRAIN_SPLIT, TEST_SPLIT, PATIENT_NSR, PATIENT_AF, RECORD_AF
from model_narin_knn_1 import MODEL_NARIN_KNN_1
from model_narin_knn_3 import MODEL_NARIN_KNN_3
from util import metrics, physionet_util
from visualization import plot_results

RUN_MODEL = True
RUNS = 1000
CV_VALUE = 10


def main_narin():
    for model in [MODEL_NARIN_KNN_3, MODEL_NARIN_KNN_1]:
        csv_file = Path(NARIN_RESULTS, model.name + ".csv")
        print(model.name)
        if RUN_MODEL:
            run_narin(n_knn=model.n_knn,
                      csv_file=csv_file,
                      get_features_funct=model.get_narin_features,
                      runs=RUNS,
                      cv_value=CV_VALUE)
        plot_results.create_figures(model.exp_metrics, csv_file, model.name, NARIN_FIGURES,
                                    ["accuracy", "sensitivity", "specificity"])
        print('-' * 80)


def run_narin(n_knn, csv_file, get_features_funct, runs, cv_value):
    exp_dict_1 = {"experience_name": "train - AF - cv patient",
                  "split": [TRAIN_SPLIT],
                  "patient_type": [PATIENT_AF],
                  "use_groups": True
                  }
    exp_dict_2 = {"experience_name": "train - AF",
                  "split": [TRAIN_SPLIT],
                  "patient_type": [PATIENT_AF],
                  "use_groups": False
                  }
    exp_dict_3 = {"experience_name": "train - all - cv patient",
                  "split": [TRAIN_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "use_groups": True
                  }
    exp_dict_4 = {"experience_name": "train - all",
                  "split": [TRAIN_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "use_groups": False
                  }
    exp_dict_5 = {"experience_name": "all - all - cv patient",
                  "split": [TRAIN_SPLIT, TEST_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "use_groups": True
                  }
    exp_dict_6 = {"experience_name": "all - all",
                  "split": [TRAIN_SPLIT, TEST_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "use_groups": False
                  }

    row_list = []
    for exp_dict in [exp_dict_1, exp_dict_2, exp_dict_3, exp_dict_4, exp_dict_5, exp_dict_6]:
        row_list = run_experiments(n_knn, get_features_funct, row_list, **exp_dict, runs=runs, cv_value=cv_value)

    df = pd.DataFrame(row_list)
    df.to_csv(csv_file)


def run_experiments(n_knn, get_features, row_list, experience_name, split, patient_type, use_groups, runs, cv_value):
    print(f"{experience_name}")
    print("Load dataset")
    x, y, groups = load_dataset(min_window=0, max_window=30,
                                window_size=5, step=2.5,
                                exclude=["n27"], split=split,
                                patient_type=patient_type)
    x_feat = get_features(x)

    print(f"Run experiments")
    for i in tqdm(range(runs)):
        model = KNeighborsClassifier(n_neighbors=n_knn)
        if use_groups:
            cv = StratifiedGroupKFold(n_splits=cv_value, shuffle=True)
            scores = cross_validate(model, x_feat, y, groups=groups, scoring=metrics.SCORES, cv=cv)
        else:
            cv = KFold(cv_value, shuffle=True)
            scores = cross_validate(model, x_feat, y, scoring=metrics.SCORES, cv=cv)
        row_list.append({
            "experience_name": experience_name,
            "knn": n_knn,
            "split": split,
            "patient_type": patient_type,
            "use_groups": use_groups,
            "accuracy": np.mean(scores["test_accuracy"]),
            "sensitivity": np.mean(scores["test_sensitivity"]),
            "specificity": np.mean(scores["test_specificity"])
        })
    print("-" * 80)
    return row_list


def load_dataset(min_window, max_window, window_size, step, exclude=None, split=None, patient_type=None):
    df = pd.read_csv(LABEL_FILE)
    if exclude is not None:
        df = df[~df.record_id.isin(exclude)]
    if split is not None:
        df = df[df.original_split.isin(split)]
    if patient_type is not None:
        df = df[df.patient_type.isin(patient_type)]

    patients = []
    x = []
    y = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        for i in np.arange(min_window, (max_window - window_size + step), step):
            rr = physionet_util.read_rr_from_file(row.record_id, i, i + 5)
            x.append(rr)
            y.append(1 if row.label == RECORD_AF else 0)
            patients.append(row.patient_id)
    return x, y, patients


if __name__ == '__main__':
    main_narin()
