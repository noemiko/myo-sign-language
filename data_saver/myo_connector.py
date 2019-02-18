"""Myo-to-OSC application.
Connects to a Myo, then sends EMG and IMU data as OSC messages to localhost:3000
basic on https://github.com/cpmpercussion/myo-to-osc

# Orientation data, represented as a unit quaternion. Values are multiplied by ORIENTATION_SCALE.
# Accelerometer data. In units of g. Range of + -16. Values are multiplied by ACCELEROMETER_SCALE.
# Gyroscope data. In units of deg/s. Range of + -2000. Values are multiplied by GYROSCOPE_SCALE.
# Default IMU sample rate in Hz.
DEFAULT_IMU_SAMPLE_RATE = 50
EMG_DEFAULT_STREAMING_RATE = 200

# Scale values for unpacking IMU data
ORIENTATION_SCALE = 16384.0  # See imu_data_t::orientation
ACCELEROMETER_SCALE = 2048.0  # See imu_data_t::accelerometer
GYROSCOPE_SCALE = 16.0  # See imu_data_t::gyroscope
"""
from core.myo_raw import MyoRaw
from myo.myohw import IMU_Mode
from myo.myohw import Sleep_Mode
from myo.myohw import Classifier_Mode

import argparse
from datetime import datetime
from pythonosc import udp_client

osc_client = None


def proc_imu(quat, acc, gyro):
    data_to_send = "acc,{0[0]},{0[1]},{0[2]},gyro,{1[0]},{1[1]},{1[2]},quat,{2[0]},{2[1]},{2[2]},{2[3]},{3}" \
        .format(acc, gyro, quat, datetime.now().isoformat())
    print(data_to_send)
    osc_client.send_message("/imu", data_to_send)


def proc_emg(emg_data):
    data_to_send = "emg,{0[0]},{0[1]},{0[2]},{0[3]},{0[4]},{0[5]},{0[6]},{0[7]},{1}" \
        .format(emg_data, datetime.now().isoformat())
    print(data_to_send)
    osc_client.send_message("/emg", data_to_send)


def proc_battery(battery_level):
    # print("Battery", battery_level, end='\r')
    osc_client.send_message("/battery", battery_level)


def listen_on_myo():
    # Setup Myo Connection
    # m = Myo()  # scan for USB bluetooth adapter and start the serial connection automatically
    # m = Myo(tty="/dev/tty.usbmodem1")  # MacOS
    m = MyoRaw(tty="/dev/ttyACM0")  # Linux
    m.add_emg_handler(proc_emg)
    m.add_imu_handler(proc_imu)
    m.add_battery_handler(proc_battery)

    # m.connect(address=args.address)  # connects to specific Myo unless arg.address is none.
    # Setup Myo mode, buzzes when ready.
    m.sleep_mode(Sleep_Mode.never_sleep.value)
    # EMG and IMU are enabled, classifier is disabled (thus, no sync gestures required, less annoying buzzing).
    m.set_mode(MyoRaw.send_emg.value, IMU_Mode.send_data.value, Classifier_Mode.disabled.value)
    # Buzz to show Myo is ready.
    m.vibrate(1)

    print("Now running...")
    try:
        while True:
            m.run()
    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print("\nDisconnected")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Connects to OSC server that send data from MYO')
    parser.add_argument('-a', '--address', dest='address',
                        help='Write down numbers that represent port where is data sended".')

    args = parser.parse_args()
    port_for_sending = None
    default_port = 3002
    if args.address:
        port_for_sending = args.address
    else:
        port_for_sending = default_port
    listen_on_myo()
    osc_client = udp_client.SimpleUDPClient("localhost", port_for_sending)  # OSC Client for sending messages.
