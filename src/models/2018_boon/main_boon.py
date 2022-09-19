from pathlib import Path

import hrvanalysis
import nolds
import numpy as np
import pandas as pd
import spectrum
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from CONSTANTS import BOON_RESULTS, LABEL_FILE, RECORD_AF, BOON_FIGURES
from CONSTANTS import PATIENT_NSR, PATIENT_AF, TRAIN_SPLIT, TEST_SPLIT
from features import bispectral_analysis
from util import metrics, physionet_util
from visualization import plot_results

RUN = 10
CV = 10

exp_metrics = {
    "accuracy": 0.877,
    "sensitivity": 0.868,
    "specificity": 0.887,
}


def main_boon():
    name = "boon"
    csv_file = Path(BOON_RESULTS, f"{name}.csv")
    run_boon(csv_file, runs=RUN, k_fold=CV)
    df = pd.read_csv(csv_file)
    metrics.ci_95(df, exp_metrics, ["accuracy", "sensitivity", "specificity"])
    plot_results.create_figures(exp_metrics, csv_file, name, BOON_FIGURES,
                                ["accuracy", "sensitivity", "specificity"])


def run_boon(csv_file, runs, k_fold):
    exp_dict_1 = {
        "experience_name": "TRAIN+TEST - last - AF - cv patient",
        "windows_choice": "last",
        "split": [TRAIN_SPLIT, TEST_SPLIT],
        "patient": [PATIENT_AF],
    }
    exp_dict_2 = {
        "experience_name": "TRAIN+TEST - last - AF+NSR - cv patient",
        "windows_choice": "last",
        "split": [TRAIN_SPLIT, TEST_SPLIT],
        "patient": [PATIENT_AF, PATIENT_NSR],
    }
    exp_dict_3 = {
        "experience_name": "TRAIN+TEST - all - AF - cv patient",
        "windows_choice": "all",
        "split": [TRAIN_SPLIT, TEST_SPLIT],
        "patient": [PATIENT_AF],
    }
    exp_dict_4 = {
        "experience_name": "TRAIN+TEST - all - AF+NSR - cv patient",
        "windows_choice": "all",
        "split": [TRAIN_SPLIT, TEST_SPLIT],
        "patient": [PATIENT_AF, PATIENT_NSR],
    }

    row_list = []
    for exp_dict in [exp_dict_1, exp_dict_2, exp_dict_3, exp_dict_4]:
        row_list = run_experiments(row_list, **exp_dict, runs=runs, k_fold=k_fold)
    df = pd.DataFrame(row_list)
    df.to_csv(csv_file)


def run_experiments(row_list, experience_name, windows_choice, split, patient, runs, k_fold):
    print(f"{experience_name}")
    print("Load dataset")
    if windows_choice == "last":
        windows = [[25, 30]]
    elif windows_choice == "all":
        windows = []
        for i in np.arange(0, 30 - 5 + 2.5, 2.5):
            windows.append([i, i + 5])
    else:
        raise ValueError("Unexpected windows choice")
    x, y, groups = load_dataset(list_windows=windows,
                                split=split,
                                patient_type=patient)

    C_values = [0.1, 1, 10, 100, 200, 500, 1000, 10000]
    gamma_values = [10, 5, 3, 1, 0.1, 0.01, 0.001, 0.0001]

    print(f"Run experiments")
    for i in tqdm(range(runs)):
        cv = StratifiedGroupKFold(k_fold, shuffle=True)  # cv with patient group
        C = np.random.choice(C_values)
        gamma = np.random.choice(gamma_values)
        pipe = Pipeline([
            # ("norm", Normalizer(norm="l2")),
            ("norm", StandardScaler()),
            ("svm", SVC(C=C, kernel='rbf', gamma=gamma))]
        )
        scores = cross_validate(pipe, x, y, groups=groups, scoring=metrics.SCORES, cv=cv)
        row_list.append({
            "experience_name": experience_name,
            "windows_choice": windows_choice,
            "split": split,
            "patient_type": patient,
            "accuracy": np.mean(scores["test_accuracy"]),
            "sensitivity": np.mean(scores["test_sensitivity"]),
            "specificity": np.mean(scores["test_specificity"])
        })
    print("-" * 80)

    return row_list


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
            rr_features = boon_features(rr)

            x.append(rr_features)
            y.append(1 if row.label == RECORD_AF else 0)
            patients.append(row.patient_id)
    return x, y, patients


def boon_features(rrs):
    """HRV CORRECTION"""
    nns = rrs.copy()
    nns = hrvanalysis.remove_outliers(nns, low_rri=200, high_rri=2000, verbose=False)
    nns = pd.Series(nns).interpolate().tolist()
    nns = hrvanalysis.remove_ectopic_beats(nns, method='custom', custom_removing_rule=0.3, verbose=False)
    nns = pd.Series(nns).interpolate().interpolate(limit_direction='backward').to_numpy()
    """
    selected features as described in Discussion
    NN50, pNN50, SampEn, SD2, AR-LF, LL-H1, ROI-WCOB(f2m)
    """
    diff_nni = np.diff(nns)
    nn_50 = sum(np.abs(diff_nni) > 50)
    pnn_50 = 100 * nn_50 / len(nns)

    sampen = nolds.sampen(nns, emb_dim=2, tolerance=0.38 * np.std(nns, ddof=1))

    pp = hrvanalysis.get_poincare_plot_features(nns)
    sd2 = pp['sd2']

    # spectral_features = spectral_analysis.get_frequency_domain_features(
    #     rrs.copy(),
    #     method=spectral_analysis.AR_BURG_METHOD,
    #     sampling_frequency=7.0,
    #     interpolation_method=spectral_analysis.LINEAR_INTER,
    #     detrend_method=spectral_analysis.CONSTANT_DETREND,
    #    )
    # lf_ar = spectral_features['lf']

    ar_order = 16
    nn_intervals = nns - np.mean(nns)
    # https://stackoverflow.com/questions/67278527/python-spectrums-burg-algorithm-and-plotting
    psd = np.array(spectrum.pburg(nn_intervals, ar_order, NFFT=4096).psd)
    freq = np.array(spectrum.pburg(nn_intervals, ar_order, NFFT=4096).frequencies())
    lf_band = (0.04, 0.15)
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])

    # ========================
    # Bispec features
    # LL-H1, ROI-WCOB(f2m)
    # Heart rate correction enabled (not precised how?)
    # detrending - contant
    # interpolation - cubic spline
    # resampling - 7hz
    # window func - blackman
    # seg size - 256
    # zero padding size for the segmented records - 512
    # records overlapping - 0%

    bispectral_features = bispectral_analysis.get_bispectral_features(
        nns,
        nlag=256, nsamp=256, overlap=0,
        flag='biased', nfft=512, wind=None,
        normalize=False)

    ll_h1 = bispectral_features['ll_h1']
    roi_wcob_f2 = bispectral_features['roi_wcob_f2']

    return [nn_50, pnn_50, sampen, sd2, lf, ll_h1, roi_wcob_f2]


if __name__ == '__main__':
    main_boon()
