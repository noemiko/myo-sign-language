import pandas as pd
from pytest import fixture

from myo_sign_language.data_processing.files_processing import process_file_by_params


@fixture
def test_data():
    return \
        {"timestamp": ["17:19:15.640359", "17:19:15.640759", "17:19:15.640759"],
         "orientation1": ["6030", "-14338", "-14336"],
         "orientation2": ["-4862", "1692", "1698"],
         "orientation3": ["6030", "6030", "6030"],
         "orientation4": ["-4862", "-4861", "-4861"],
         "gyro1": ["-16", "-22", "-13"],
         "gyro2": ["-2", "-2", "1"],
         "gyro3": ["2", "5", "5"],
         "acc1": ["1132", "1141", "1137"],
         "acc2": ["-780", "-796", "-798"],
         "acc3": ["1370", "1355", "1372"],
         "emg1": ["-2", "-1", "0"],
         "emg2": ["-1", "1", "-3"],
         "emg3": ["-1", "1", "-3"],
         "emg4": ["-1", "1", "-3"],
         "emg5": ["-1", "1", "-3"],
         "emg6": ["-1", "1", "-3"],
         "emg7": ["-1", "1", "-3"],
         "emg8": ["-1", "1", "-3"], }


def test_process_file_by_params_return_the_same_data_without_date_and_with_numeric_values(test_data):
    frame_from_dict = pd.DataFrame.from_dict(test_data)

    frame_without_time = frame_from_dict.drop(labels=['timestamp'], axis=1)
    numeric_test_frame = frame_without_time.apply(pd.to_numeric)

    tested_frame = process_file_by_params(frame_from_dict)
    assert tested_frame.equals(numeric_test_frame)
