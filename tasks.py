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
def visualize_a_batch(c, filename="src/configs/config.yaml", use_val=False):
    from src.visualize.visualize_batch import visualize_a_batch

    visualize_a_batch(filename, use_val)


@task
def visualize_default_box(c, filename="src/configs/config.yaml"):
    from src.visualize.visualize_default_box import visualize_default_box

    visualize_default_box(filename)


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
