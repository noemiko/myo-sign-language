"""Small example OSC server

This program listens on defined port and save data to disk.
"""
import argparse
from datetime import datetime
import csv
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server


class Saver(object):
    def __init__(self):
        """
        Save data from osc client.

        """
        self.server = ""
        self.sign_data = []
        self.timer_2sec = ''
        self.file_name = ""
        self.ip = "127.0.0.1"
        self.port = None
        self.file_counter = 0

    def start_listen(self, port):
        """
        :param port: port which one this listener wait for data
        :return:
        """
        self.port = port
        self.start_server_to_listen(self.user_callback_imu, self.user_callback_emg)

    def start_new_record(self):
        print(" Ready?")
        name = input("Give me file name/n")
        self.file_name = str(datetime.now()) + name
        input("Press Enter to continue...")
        self.sign_data = []
        self.timer_2sec = time.time() + 2

    def user_callback_imu(self, address_come_from, args):
        self.save_raw_data(args)

    def user_callback_emg(self, address_come_from, args):
        self.save_raw_data(args)

    def save_raw_data(self, myo_data):
        row = self.split_with_timestamp(myo_data)
        self.sign_data.append(row)
        if time.time() >= self.timer_2sec:
            new_file_name = "[" + str(self.file_counter) + "]_" + self.file_name + ".csv"
            with open("./data/" + new_file_name, 'w') as new_csv_file:
                self.save_file(new_csv_file, self.sign_data)
            print("Created new file", new_file_name)
            # self.server.server_close()
            self.sign_data = []
            self.start_new_record()

    def save_file(self, new_csv_file, data):
        sensor_writer = csv.writer(new_csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in data:
            sensor_writer.writerow(row)

    def split_with_timestamp(self, myo_data):
        """
        Change string separated with delimiter
        to list with timestamp[2018-03-19T23:03:47.100725, yolo, 123, 345]
        """
        raw_data = myo_data.split(",")
        raw_data.insert(0, datetime.now().isoformat())
        return raw_data

    def start_server_to_listen(self, imu_handler, emg_handler):
        self.file_counter += 1
        dispatcher = Dispatcher()
        dispatcher.map("/imu", imu_handler)
        dispatcher.map("/emg", emg_handler)
        self.server = osc_server.OSCUDPServer((self.ip, self.port), dispatcher)
        print("Serving on {}".format(self.server.server_address))
        self.server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Connects to OSC server that send data from MYO')
    parser.add_argument('-a', '--address', dest='address',
                        help='Write down numbers that represent port where is data sended".')

    args = parser.parse_args()
    port_to_listen = None
    default_port = 3002
    if args.address:
        port_to_listen = args.address
        print("Get data from user port {}".format(args.address))
    else:
        port_to_listen = default_port
    saver = Saver().start_listen(port_to_listen)
    saver.start_new_record()
