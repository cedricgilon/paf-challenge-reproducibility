import numpy as np
import hrvanalysis


class MODEL_NARIN_KNN_3:
    n_knn = 3
    name = f"narin_knn_{n_knn}"
    exp_metrics = {
        "sensitivity": 0.92,
        "specificity": 0.88,
        "neg": 0.91,
        "pos": 0.88,
        "accuracy": 0.90
    }

    @staticmethod
    def get_narin_features(x):
        x_feat = []
        for rr in x:
            diff_nni = np.diff(rr)
            rmssd = np.sqrt(np.mean(diff_nni ** 2))

            frequential_freatures = hrvanalysis.get_frequency_domain_features(nn_intervals=rr,
                                                                              method="welch",
                                                                              sampling_frequency=7,
                                                                              interpolation_method="cubic")

            fft_vlf = frequential_freatures["vlf"]
            fft_lf = frequential_freatures["lf"]
            fft_total = frequential_freatures["total_power"]

            x_feat.append([rmssd, fft_vlf, fft_lf, fft_total])
        return x_feat
