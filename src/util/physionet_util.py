from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb.processing

from CONSTANTS import PHYSIONET_SAMPLING_FREQUENCY, RECORDS_FOLDER, LABEL_FILE, RECORD_AF


def read_rr_from_file(record_id, window_start, window_stop, is_milliseconds=True):
    start_index = int(window_start * 60 * PHYSIONET_SAMPLING_FREQUENCY)
    stop_index = int(window_stop * 60 * PHYSIONET_SAMPLING_FREQUENCY)
    assert start_index < stop_index
    record_path = Path(RECORDS_FOLDER, record_id, record_id)
    record = wfdb.rdrecord(str(record_path), sampfrom=start_index, sampto=stop_index)
    ann = wfdb.rdann(str(record_path), 'qrs', sampfrom=start_index, sampto=stop_index)
    qrs_locs = ann.sample
    rr = wfdb.processing.hr.calc_rr(qrs_locs, record.fs, rr_units='seconds')
    if is_milliseconds:
        rr = rr * 1000
    return rr
