import numpy as np


class MODEL_NARIN_KNN_1:
    n_knn = 1
    name = f"narin_knn_{n_knn}"
    exp_metrics = {
        "sensitivity": 0.84,
        "specificity": 0.84,
        "neg": 0.84,
        "pos": 0.84,
        "accuracy": 0.84
    }

    @staticmethod
    def get_narin_features(x):
        x_feat = []
        for rr in x:
            sdnn = np.std(rr)
            diff_nni = np.diff(rr)
            nn50 = sum(np.abs(diff_nni) > 50)
            x_feat.append([sdnn, nn50])
        return x_feat
