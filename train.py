import argparse
import yaml

def load_configurations():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args()

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    return config

def main():

    config = load_configurations()

    print(config)


if __name__ == "__main__":

    main()
