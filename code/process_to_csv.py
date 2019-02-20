import os
import csv
import os, errno
import math
import time
from datetime import datetime
from dateutil.parser import parse


def toEulerAngle(w, x, y, z):
    """ Quaternion to Euler angle conversion borrowed from wikipedia.
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles """
    # roll (x-axis rotation)
    sinr = +2.0 * (w * x + y * z)
    cosr = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # pitch (y-axis rotation)
    sinp = +2.0 * (w * y - z * x)
    if math.fabs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny = +2.0 * (w * z + x * y)
    cosy = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def process_to_readeable_csv_format():
    """
    Process csv files with MYO data for format that is easier to process.

    change from:
    "2018-03-27T16:02:39.416143","acc","-310","-1201","-1828","gyro","-29","0","4","quat","-3936","14663","5532","-2705","2018-03-27T16:02:39.415082"
    "2018-03-27T16:02:39.417310","emg","-1","2","1","1","-1","-8","-1","-1","2018-03-27T16:02:39.416490"
    "2018-03-27T16:02:39.417617","emg","-1","-2","0","0","-1","-1","-2","-1","2018-03-27T16:02:39.416907"
    "2018-03-27T16:02:39.422174","emg","1","1","-1","-1","1","-2","0","-1","2018-03-27T16:02:39.421249"
    "2018-03-27T16:02:39.422482","emg","1","-1","0","-1","-1","0","-2","0","2018-03-27T16:02:39.421705"

    to:
    timestamp, acc1, acc2, acc3, gyro1, gyro2, gyro3, orientation1, orientation2, orientation3, orientation4, emg1, emg2, emg3, emg4, emg5,emg6,emg7,emg8
    "2018-03-27T16:02:39.416143","-310","-1201","-1828","-29","0","4","-3936","14663","5532","-2705", "-1","2","1","1","-1","-8","-1","-1"
    "2018-03-27T16:02:39.416143","-310","-1201","-1828","-29","0","4","-3936","14663","5532","-2705", "-1","-2","0","0","-1","-1","-2","-1"
    "2018-03-27T16:02:39.416143","-310","-1201","-1828","-29","0","4","-3936","14663","5532","-2705", "1","1","-1","-1","1","-2","0","-1"

    myo send OMI data with 50hz and EMG with 200hz,
    data from OMI will be duplicated for each row with EMG that will be received earlier than next OMI data

    :return:
    """
    data_folder = 'data/raw_data'
    for root, dirs, files in os.walk(data_folder):
        root_folder = root.split("/")[-1]
        new_dir = "data/processed_imu_data/{}".format(root_folder)
        create_the_same_folder_name_as_source(new_dir)
        for name in files:
            file_path = os.path.join(root, name)
            processed_file = []
            print("Opening file:", file_path)
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
                processed_file = create_proper_file_format(reader)
                print(processed_file)
            new_file_path = os.path.join(new_dir, name)
            print(new_file_path)

            with open(new_file_path, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
                schema = ["timestamp", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3", "orientation1",
                          "orientation2", "orientation3", "orientation4", "emg1", "emg2", "emg3", "emg4", "emg5",
                          "emg6", "emg7", "emg8"]
                writer.writerow([g for g in schema])
                for row in processed_file:
                    print(row)
                    writer.writerow(row)


def create_proper_file_format(rows):
    """
    timestamp, acc1, acc2, acc3, gyro1, gyro2, gyro3,
    orientation1, orientation2, orientation3, orientation4,
    emg1, emg2, emg3, emg4, emg5, emg6, emg7, emg8
    """
    processed_file = []
    rows = list(rows)
    for index, row in enumerate(rows):
        print(row)
        if row[1] == "acc":
            new_row = ['']  # timestamp
            new_row.append(row[2])  # acc1,
            new_row.append(row[3])  # acc2
            new_row.append(row[4])  # acc3
            new_row.append(row[6])  # gyro1
            new_row.append(row[7])  # gyro2
            new_row.append(row[8])  # gyro3
            new_row.append(row[10])
            new_row.append(row[11])
            new_row.append(row[12])
            new_row.append(row[13])
            #  new_row.extend(toEulerAngle(float(row[10]), float(row[11]), float(row[12]), float(row[13])))  # orient

            for i in range(-2, 3):
                if i == 0:
                    continue
                elif index == 0 or index == 1:
                    continue
                try:
                    emg_data = get_processed_emg(rows[index + i])
                except IndexError:
                    if index + i > 490:
                        continue
                    else:
                        raise
                timestamp = emg_data.pop(0)
                # "2018-04-16T19:49:04.765537"
                parsed_timestamp = parse(timestamp).time()
                new_row[0] = parsed_timestamp
                print(new_row)
                processed_file.append(new_row + emg_data)
    return processed_file


def get_processed_emg(emg_row):
    for index, emg in enumerate(emg_row):
        if index == 0 or index == 1 or index == 10:
            continue
        emg_row[index] = emg
    return [emg_row[0], emg_row[2], emg_row[3], emg_row[4], emg_row[5], emg_row[6], emg_row[7], emg_row[8], emg_row[9]]


def create_the_same_folder_name_as_source(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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
    process_to_readeable_csv_format()
