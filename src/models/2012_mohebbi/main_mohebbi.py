from pathlib import Path

import hrvanalysis
import nolds
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from CONSTANTS import MOHEBBI_RESULTS, MOHEBBI_FIGURES, LABEL_FILE
from CONSTANTS import TRAIN_SPLIT, TEST_SPLIT, PATIENT_AF, PATIENT_NSR, RECORD_AF
from features import bispectral_analysis, spectral_analysis
from util import metrics, physionet_util
from visualization import plot_results

RUNS = 1000
RANDOM_C_GAMMA = True

exp_metrics = {
    "accuracy": -1,
    "sensitivity": 0.963,
    "specificity": 0.931,
    "ppv": 0.9286
}


def main():
    if RANDOM_C_GAMMA:
        name = "mohebbi_random"
    else:
        name = "mohebbi_1000_3-6"
    csv_file = Path(MOHEBBI_RESULTS, f"{name}.csv")
    run_mohebbi(csv_file, RUNS, RANDOM_C_GAMMA)
    df = pd.read_csv(csv_file)

    metrics_choice = ["accuracy", "sensitivity", "specificity"]
    metrics.ci_95(df, exp_metrics, metrics_choice)
    plot_results.create_figures(exp_metrics, csv_file, name, MOHEBBI_FIGURES, metrics_choice)


def run_mohebbi(csv_file, runs, random_c_gamma):
    exp_dict_1 = {"experience_name": "train - AF - cv patient - no norm",
                  "train_split": [TRAIN_SPLIT],
                  "test_split": [TEST_SPLIT],
                  "patient_type": [PATIENT_AF],
                  "is_norm": False}
    exp_dict_2 = {"experience_name": "train - AF - cv patient - norm",
                  "train_split": [TRAIN_SPLIT],
                  "test_split": [TEST_SPLIT],
                  "patient_type": [PATIENT_AF],
                  "is_norm": True}
    exp_dict_3 = {"experience_name": "train - all - cv patient - no norm",
                  "train_split": [TRAIN_SPLIT],
                  "test_split": [TEST_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "is_norm": False}
    exp_dict_4 = {"experience_name": "train - all - cv patient - norm",
                  "train_split": [TRAIN_SPLIT],
                  "test_split": [TEST_SPLIT],
                  "patient_type": [PATIENT_AF, PATIENT_NSR],
                  "is_norm": True}

    row_list = []
    for exp_dict in [exp_dict_1, exp_dict_2, exp_dict_3, exp_dict_4]:
        row_list = run_experiments(row_list, **exp_dict, runs=runs, random_c_gamma=random_c_gamma)

    df = pd.DataFrame(row_list)
    df.to_csv(csv_file)

    return csv_file


def run_experiments(row_list, experience_name, train_split, test_split, patient_type, is_norm, runs, random_c_gamma):
    print(f"{experience_name}")
    print("Load dataset")
    windows = [[0, 30]]

    x_train, y_train, groups = load_dataset(list_windows=windows,
                                            split=train_split,
                                            patient_type=patient_type)
    x_train = [mohebbi_features(_x) for _x in tqdm(x_train)]
    x_test, y_test, groups = load_dataset(list_windows=windows,
                                          split=test_split,
                                          patient_type=patient_type)
    x_test = [mohebbi_features(_x) for _x in tqdm(x_test)]

    if is_norm:
        normalizer = StandardScaler()
        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)

    if random_c_gamma:
        C_values = [0.1, 1, 10, 100, 200, 500, 1000, 10000]
        gamma_values = [10, 5, 3, 1, 0.1, 0.01, 0.001, 0.0001]
    else:
        C = 1000
        gamma = 3.6

    print(f"Run experiments")
    for i in tqdm(range(runs)):
        if random_c_gamma:
            C = np.random.choice(C_values)
            gamma = np.random.choice(gamma_values)
        model = SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        scores = {}
        for score_name, score_func in metrics.SCORES_FUNCT.items():
            scores[f'test_{score_name}'] = score_func(y_true=y_test, y_pred=y_pred)

        row_list.append({
            "experience_name": experience_name,
            "C": C,
            "gamma": gamma,
            "accuracy": np.mean(scores["test_accuracy"]),
            "ppv": np.mean(scores["test_ppv"]),
            "sensitivity": np.mean(scores["test_sensitivity"]),
            "specificity": np.mean(scores["test_specificity"]),
            "normalize": is_norm
        })
    print("-" * 80)
    return row_list


def mohebbi_features(rrs):
    # Spectral features, HF / LF
    # Mohebbi only uses constant detrending apparently and no resampling...

    spectral_features = spectral_analysis.get_frequency_domain_features(
        rrs, method='burg',
        sampling_frequency=1,
        interpolation_method=None,
        detrend_method='constant',
        ar_order=16)

    lf = spectral_features["lf"]
    hf = spectral_features["hf"]

    # Bispectral features
    # No signal detrending nor resampling
    bispectral_features = bispectral_analysis.get_bispectral_features(
        rrs,
        nlag=64, nsamp=64, overlap=0,
        flag='biased', nfft=128, wind=None,
        normalize=False)
    e1 = bispectral_features["bispen"]
    e2 = bispectral_features["bispen2"]
    h1 = bispectral_features["h1"]
    h2 = bispectral_features["h2"]
    h3 = bispectral_features["h3"]
    h4 = bispectral_features["h4"]

    # Sampentropy extracted by Mohebby has embedding n=2 and tolerance = 0.2 * std(rrs) which is
    # the default used by this library
    sampen = nolds.sampen(rrs)

    poincare_features = hrvanalysis.get_poincare_plot_features(rrs)
    ratio_sd1_sd2 = poincare_features["sd1"] / poincare_features["sd2"]

    # features = {**spectral_features, **bispectral_features, **{'sampen': sampen}, **poincare_features}
    features = [lf, hf,
                e1, e2, h1, h2, h3, h4,
                sampen,
                poincare_features["sd1"], poincare_features["sd2"],
                ratio_sd1_sd2]
    return features


def load_dataset(list_windows, exclude=None, split=None, patient_type=None):
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
        for (start, stop) in list_windows:
            rr = physionet_util.read_rr_from_file(row.record_id, start, stop)
            x.append(rr)
            y.append(1 if row.label == RECORD_AF else 0)
            patients.append(row.patient_id)
    return x, y, patients


if __name__ == '__main__':
    main()
