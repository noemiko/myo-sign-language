import os
import pandas as pd
# from myo_sign_language.data_processing.preprocessors import fourier
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set

def process_from_files(filename):
    """
    Get data from all files defined in DATA FOLDER and
    sort them by class
    :return: dict[list, list]
    """
    sorted_by_class_data = dict()

    data_path = os.path.realpath(os.path.join(__file__, '..', '..', 'data', filename))
    for root, dirs, files in os.walk(data_path):
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
                           acc_process=None)->List[List[int]]:
    """

    :param file:
    :param orientation_process:
    :param gyro_process:
    :param emg_process:
    :param acc_process:
    :return: List[List[int]] Data separated for each sensor (max 18)
    """
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


    concatenated = pd.concat(to_concat, sort=False, axis=1)
    return concatenated.values.transpose().tolist()


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
