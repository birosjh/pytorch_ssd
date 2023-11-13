from invoke import task

@task
def train(c, filename="src/configs/config.yaml"):

    from src.scripts.train import train_model

    train_model(filename)


@task
def infer(c, image_filename, output_filename, filename="src/configs/config.yaml"):

    from src.scripts.inference import inference

    inference(filename, image_filename, output_filename)


@task
def visualize_a_batch(c, filename="configs/config.yaml", use_val=False):

    from src.visualize.visualize_batch import visualize_batch

    visualize_batch(filename, use_val)


@task
def lint(c, filename="."):
    c.run(f"ruff check {filename}")


@task
def format(c, filename="."):
    c.run(f"ruff format {filename}")


@task
def test(c, filename="*"):
    c.run(f"python3 -m unittest tests/test_{filename}.py")


@task
def create_test_data(c):
    c.run("python3 scripts/prepare_test_data.py")
