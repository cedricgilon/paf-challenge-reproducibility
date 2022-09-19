from pathlib import Path

# PHYSIONET

PHYSIONET_SAMPLING_FREQUENCY = 128
PATIENT_AF = "patient_af"
PATIENT_NSR = "patient_nsr"
RECORD_NSR = "record_nsr"
RECORD_AF = "record_af"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# FILES

this_file = Path(__file__)
PROJECT_PATH = this_file.parent.parent
DATA_FOLDER = Path(PROJECT_PATH, "data")
LABEL_FILE = Path(DATA_FOLDER, "physionet_afpdb.csv")
RECORDS_FOLDER = Path(DATA_FOLDER, "records")

# MOHEBBI 2012
# Prediction of paroxysmal atrial fibrillation based on non-linear analysis and spectrum and bispectrum features of
# the heart rate variability signal

MOHEBBI = Path(PROJECT_PATH, "src", "models", "2012_mohebbi")
MOHEBBI_FIGURES = Path(MOHEBBI, "figures")
MOHEBBI_RESULTS = Path(MOHEBBI, "results")

# BOON 2018
# Paroxysmal atrial fibrillation prediction based on HRV analysis and non-dominated sorting genetic algorithm III

BOON = Path(PROJECT_PATH, "src", "models", "2018_boon")
BOON_FIGURES = Path(BOON, "figures")
BOON_RESULTS = Path(BOON, "results")

# NARIN 2018
# Early prediction of paroxysmal atrial fibrillation based on short-term heart rate variability

NARIN = Path(PROJECT_PATH, "src", "models", "2018_narin")
NARIN_FIGURES = Path(NARIN, "figures")
NARIN_RESULTS = Path(NARIN, "results")
