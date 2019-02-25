import os
import pandas as pd
from myo_sign_language.data_processing.preprocessors import fourier

DATA_FOLDER = '../data/one_line_per_all_sensors'


def process_from_files():
    """
    Get data from all files defined in DATA FOLDER and
    sort them by class
    :return: dict[list, list]
    """
    sorted_by_class_data = dict()
    for root, dirs, files in os.walk(DATA_FOLDER):
        for name in files:
            file_path = os.path.join(root, name)
            file_array = pd.read_csv(file_path)
            move_index = get_move_index(name)
            processed_file = process_file_by_params(file_array)
            if move_index in sorted_by_class_data:
                sorted_by_class_data[move_index].append(processed_file)
            else:
                sorted_by_class_data[move_index] = [processed_file]
    print('end sort data')
    return sorted_by_class_data


def process_file_by_params(file,
                           orientation_process=None,
                           gyro_process=None,
                           emg_process=None,
                           acc_process=None):
    file_without_time = file.drop(labels=['timestamp'], axis=1)
    numeric_file = file_without_time.apply(pd.to_numeric)

    orientation_labels = ["orientation1", "orientation2", "orientation3", "orientation4"]
    gyro_labels = ["gyro1", "gyro2", "gyro3"]
    acc_labels = ["acc1", "acc2", "acc3"]
    emg_labels = ["emg1", "emg2", "emg3", "emg4", 'emg5', 'emg6', "emg7", "emg8"]

    numeric_orientation = numeric_file[orientation_labels]
    numeric_gyro = numeric_file[gyro_labels]
    numeric_acc = numeric_file[acc_labels]
    numeric_emg = numeric_file[emg_labels]

    to_concat = []

    def concat_process(dataframe, process):
        if process is not None:
            dataframe = dataframe.apply(process)
        to_concat.append(dataframe)

    concat_process(numeric_orientation, orientation_process)
    concat_process(numeric_gyro, gyro_process)
    concat_process(numeric_acc, acc_process)
    concat_process(numeric_emg, emg_process)

    return pd.concat(to_concat, sort=False, axis=1)


def get_move_index(name):
    """
    Files with myo data has format:
    [fileNumber]YYYY-MM-dd hh:mm:ss.miliseconds_MoveNumber.csv
    """
    move_index_with_extension = name.split('_')[-1]
    move_index = move_index_with_extension.split('.')[0]
    valid_move = validate_move_number(move_index)

    return valid_move


def validate_move_number(move_index):
    try:
        int(move_index)
    except ValueError:
        raise ValueError(f"move should be a number: {move_index}")
    if int(move_index) not in range(19):
        raise ValueError(f"Wrong move number {move_index}. Is should be between 1 nad 18")
    return move_index


if __name__ == "__main__":
    # process_from_files()
    test_data = {"timestamp": ["17:19:15.640359", "17:19:15.640759", "17:19:15.640759"],
                 "acc1": ["1132", "1141", "1137"],
                 "acc2": ["-780", "-796", "-798"],
                 "acc3": ["1370", "1355", "1372"],
                 "gyro1": ["-16", "-22", "-13"],
                 "gyro2": ["-2", "-2", "1"],
                 "gyro3": ["2", "5", "5"],
                 "orientation1": ["6030", "-14338", "-14336"],
                 "orientation2": ["-4862", "1692", "1698"],
                 "orientation3": ["6030", "6030", "6030"],
                 "orientation4": ["-4862", "-4861", "-4861"],
                 "emg1": ["-2", "-1", "0"],
                 "emg2": ["-1", "1", "-3"],
                 "emg3": ["-1", "1", "-3"],
                 "emg4": ["-1", "1", "-3"],
                 "emg5": ["-1", "1", "-3"],
                 "emg6": ["-1", "1", "-3"],
                 "emg7": ["-1", "1", "-3"],
                 "emg8": ["-1", "1", "-3"], }

    test_frame = pd.DataFrame.from_dict(test_data)
    print(process_file_by_params(test_frame))
