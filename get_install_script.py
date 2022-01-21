from gettext import install
import sys
import yaml

ml = sys.argv[1]

# Load YAML file:
with open('algorithms.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

algorithm = [alg for alg in config['algorithms'] if alg['name'] == ml][0]

install_scripts = []

if 'install' in config.keys():
    install_scripts.extend(config['install'])

if 'install' in algorithm.keys():
    install_scripts.extend(algorithm['install'])

print(' '.join(install_scripts))