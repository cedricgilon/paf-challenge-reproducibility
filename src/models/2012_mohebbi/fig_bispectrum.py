from features import bispectral_analysis
from util.physionet_util import read_rr_from_file


def main_bispectral():
    # rr = read_rr_from_file("p03", 0, 30)
    rr = read_rr_from_file("p16", 0, 30, is_milliseconds=False)
    # rr = read_rr_from_file("p19", 0, 30)

    bispectral_features = bispectral_analysis.get_bispectral_features(
        rr,
        nlag=64, nsamp=64, overlap=0,
        flag='biased', nfft=128, wind=None,
        normalize=False,
        plot=True)


if __name__ == '__main__':
    main_bispectral()
