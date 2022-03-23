from invoke import task


@task
def lint(c, filename="."):

    c.run(f"flake8 {filename}")


@task
def test(c, filename="*"):

    c.run(f"python -m unittest tests/test_{filename}.py")


@task
def format(c, filename="."):

    c.run(f"isort {filename}")
    c.run(f"black {filename}")
    c.run(f"flake8 {filename}")
