import os
import argparse
import configparser

from run import run

parser = argparse.ArgumentParser()

parser.add_argument('--names', nargs='*', required=True)
parser.add_argument('-e', '--end', type=int, default=50000)
parser.add_argument('-s', '--step', type=int, default=5000)
parser.add_argument('-g', '--gif', action='store_true')

args = parser.parse_args()

start = args.step if args.step > 0 else 5000
end = args.end if args.end >= args.step else 50000
step = start
export_gifs = args.gif

for i in range(start, end + 1, step):
    for name in args.names:
        config_path = "training/" + name + "/config.ini"
        config_dir = os.path.dirname(config_path)

        config = configparser.ConfigParser()
        config.read(config_path)
        checkpoint = int(config['DEFAULT'].get('checkpoint', '0'))

        if checkpoint >= i:
            continue

        config['criteria']['max_episodes'] = str(i)

        with open(config_path, 'w') as configfile:
            config.write(configfile)

        print(f"Running test {name} to episode {i}")
        run(name, 0, export_gifs, False)
