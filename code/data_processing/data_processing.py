import os
import pandas as pd
from data_processing.preprocessors import remove_timestamp

DATA_FOLDER = 'duplicated_imu_data'


def process_from_files():
    """
    Get data from all files defined in DATA FOLDER and
    sort them by class
    :return: dict[list, list]
    """
    sorted_by_class_data = dict()
    for root, dirs, files in os.walk(DATA_FOLDER, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_array = pd.read_csv(file_path)
            class_name = get_move_number(name)
            file_list_separated_by_sensors = process_file_by_params(file_array)
            if class_name in sorted_by_class_data:
                sorted_by_class_data[class_name].append(file_list_separated_by_sensors)
            else:
                sorted_by_class_data[class_name] = [file_list_separated_by_sensors]
    print('end sort data')
    return sorted_by_class_data


def process_file_by_params(file_array):
    file_list = file_array
    file_list = remove_timestamp(file_list)
    all_prepro = []
    orientation_sensor = file_list[["orientation1", "orientation2", "orientation3", "orientation4"]]
    for i in orientation_sensor.transpose().values:
        # prepossess orientation data here
        all_prepro.append(i)

    gyro_sensor = file_list[["gyro1", "gyro2", "gyro3"]]
    for i in gyro_sensor.transpose().values:
        # prepossess gyro data here
        all_prepro.append(i)

    acc_sensor = file_list[["acc1", "acc2", "acc3"]]
    for i in acc_sensor.transpose().values:
        # prepossess accelerometer data here
        all_prepro.append(i)

    emg_sensor = file_list[["emg1", "emg2", "emg3", "emg4", 'emg5', 'emg6', "emg7", "emg8"]]
    for i in emg_sensor.transpose().values:
        # prepossess accelerometer data here
        all_prepro.append(i)
    return all_prepro


def get_move_number(name):
    """
    Files with myo data has format:
    [number_of_data]YYYY-MM-dd hh:mm:ss.milisecondsMoveNumber.csv
    """
    splited_file_name = name.split('.')
    seconds_with_move_number = splited_file_name[1]
    if len(seconds_with_move_number) == 7:
        return seconds_with_move_number[-1]
    elif len(seconds_with_move_number) == 8:
        return seconds_with_move_number[-2:]
    else:
        raise ValueError("Wrong file name {}".format(name))


if __name__ == "__main__":
    process_from_files()
