import argparse
from myo_sign_language.data_saver.myo_connector import run as myo_listen
from myo_sign_language.data_saver.recorder import run as run_recording

default_port = 3002


def check_positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


parser = argparse.ArgumentParser(
    description='Run server to get data by OSC from MYO '
                'and server that listen on data from osc server and save it on disk')

parser.add_argument('-a', '--address', dest='address', type=check_positive_integer,
                    help=f'Write down numbers that represent port where is data sended,'
                         ' if not defined default is {default_port}')

parser.add_argument('-m', '--myo', dest='myo', action='store_true',
                    help='listen on myo data and send it to defined port in -a')

parser.add_argument('-r', '--recording', dest='recording', action='store_true',
                    help='listen on osc server which sends data to the port defined on -a')


def get_port(user_args):
    if user_args.address:
        return user_args.address
    else:
        return default_port


def run_server_by_params(port, user_args):
    if user_args.myo:
        try:
            myo_listen(port)
        except OSError as error:
            print(f"Please use another port than `{port}` to serve data from myo. Because this one is busy.")
    elif user_args.recording:
        try:
            run_recording(port)
        except OSError as error:
            print(f"Please use another port than {port}"
                  f" to listen on data from myo served by osc server. Because this one is busy.")
    else:
        print('Missing definition which one script you would like to run')
        parser.print_help()


if __name__ == '__main__':
    user_args = parser.parse_args()
    port = get_port(user_args)

    run_server_by_params(port, user_args)
