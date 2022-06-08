from invoke import task

from scripts.train import train_model
from visualize.visualize_batch import visualize_batch


@task
def train(c, filename="configs/config.yaml"):

    train_model(filename)


@task
def visualize_a_batch(c, filename="configs/config.yaml", use_val=False):

    visualize_batch(filename, use_val)


@task
def lint(c, filename="."):

    c.run(f"flake8 {filename}")


@task
def format(c, filename="."):

    c.run(f"isort {filename}")
    c.run(f"black {filename}")
    c.run(f"flake8 {filename}")
    c.run(f"mypy {filename}")


@task
def test(c, filename="*"):

    c.run(f"python3 -m unittest tests/test_{filename}.py")


@task
def create_test_data(c):

    c.run("python3 scripts/prepare_test_data.py")
