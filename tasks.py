from invoke import task


@task
def train(c, filename="configs/config.yaml"):

    c.run(f"python3 scripts/train.py --config {filename}")


@task
def lint(c, filename="."):

    c.run(f"flake8 {filename}")


@task
def format(c, filename="."):

    c.run(f"isort {filename}")
    c.run(f"black {filename}")
    c.run(f"flake8 {filename}")


@task
def test(c, filename="*"):

    c.run(f"python3 -m unittest tests/test_{filename}.py")


@task
def create_test_data(c):

    c.run("python3 scripts/prepare_test_data.py")
