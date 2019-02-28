import csv
import os, errno
from dateutil.parser import parse


def process_to_new_csv_format():
    """
    Process csv files with MYO data to format that is easier to process.

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

    myo send IMU data with 50hz and EMG with 200hz,
    data from IMU will be duplicated for each row with EMG that will be received earlier than next IMU data

    :return:
    """
    data_path = os.path.realpath(os.path.join(__file__, '..', '..', 'data'))

    files_to_process = os.path.join(data_path, 'raw_data')

    for root, dirs, files in os.walk(files_to_process):
        if files:
            files_folder = root.split("/")[-1]
            new_path = create_dir(files_folder, data_path)
        for name in files:
            file_path = os.path.join(root, name)
            new_file_path = os.path.join(new_path, name)
            process_file_to_new_path(file_path, new_file_path)


def process_file_to_new_path(file_dir, new_dir):
    print(f"Opening file: {file_dir}")
    content = process_file(file_dir)
    print(f"Processed file: {new_dir}")
    save_file_with_new_content(content, new_dir)


def process_file(file_dir):
    with open(file_dir, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        return create_proper_file_format(reader)


def save_file_with_new_content(content, dir):
    with open(dir, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        schema = ["timestamp", "emg1", "emg2", "emg3", "emg4", "emg5",
                  "emg6", "emg7", "emg8", "acc1", "acc2", "acc3",
                  "gyro1", "gyro2", "gyro3",
                  "orientation1", "orientation2", "orientation3", "orientation4", ]
        writer.writerow([g for g in schema])
        writer.writerows(content)
    return True


def create_proper_file_format(rows):
    """
    timestamp, acc1, acc2, acc3, gyro1, gyro2, gyro3,
    orientation1, orientation2, orientation3, orientation4,
    emg1, emg2, emg3, emg4, emg5, emg6, emg7, emg8
    """
    processed_file = []
    rows = list(rows)
    for index, row in enumerate(rows):
        if row[1] == "acc":
            acc = [row[2], row[3], row[4]]
            gyro = [row[6], row[7], row[8]]
            orientation = [row[10], row[11], row[12], row[13]]
            imu_data = acc + gyro + orientation
            emg_data_around = get_emg_data_around_imu(index, rows)
            for emg in emg_data_around:
                processed_file.append(emg + imu_data)
    return processed_file


def get_emg_data_around_imu(imu_index, file_rows):
    """
    In file may be two rows with emg data before and after imu.
    """
    emg_indexes = []
    if imu_index == 0:
        emg_indexes = [1, 2]
    elif imu_index == 1:
        emg_indexes = [-1, 1, 2]
    else:
        emg_indexes = [-2, -1, 1, 2]
    for emg_index in emg_indexes:
        try:
            yield get_formatted_emg(file_rows[imu_index + emg_index])
        except IndexError:
            pass


def get_formatted_emg(emg_row):
    """
    :param emg_row: dict [str] one row that represent data from Electromyograph sensor
    example:
    ['2018-07-04T17:39:53.743240', 'emg', '-1', '-6', '-9', '-9', '1', '1', '-1', '-2', '2018-07-04T17:39:53.742082']
    :return:
    example:
    ['2018-07-04T17:39:53.743240', 'emg', '-1', '-6', '-9', '-9', '1', '1', '-1', '-2']
    """
    new_emg_row = emg_row.copy()
    new_emg_row.pop(1)  # remove 'emg' word
    new_emg_row.pop(9)  # remove last timestamp
    return new_emg_row


def create_dir(folder_name, directory):
    dir_for_processed_files = os.path.join(directory, 'processed_imu_data')
    new_dir = os.path.join(dir_for_processed_files, folder_name)
    try:
        os.makedirs(new_dir)
        return new_dir
    except OSError as e:
        if e.errno != errno.EEXIST:
            print(e)
            raise
    return new_dir


if __name__ == "__main__":
    process_to_new_csv_format()
