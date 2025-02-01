"""
Invoke - Tasks
==============
"""

from __future__ import annotations

import contextlib
import inspect
import platform

from colour.utilities import message_box

import app

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from invoke import Context, task
from invoke.exceptions import Failure

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "APPLICATION_NAME",
    "ORG",
    "CONTAINER",
    "clean",
    "precommit",
    "tests",
    "requirements",
    "build",
    "docker_build",
    "docker_remove",
    "docker_run",
    "docker_push",
]

APPLICATION_NAME = app.__application_name__

ORG = "ampas"

CONTAINER = APPLICATION_NAME.replace(" ", "").lower()


@task
def clean(
    ctx: Context,
    docs: bool = True,
    bytecode: bool = False,
    mypy: bool = True,
    pytest: bool = True,
) -> None:
    """
    Clean the project.

    Parameters
    ----------
    ctx
        Context.
    docs
        Whether to clean the *docs* directory.
    bytecode
        Whether to clean the bytecode files, e.g. *.pyc* files.
    mypy
        Whether to clean the *Mypy* cache directory.
    pytest
        Whether to clean the *Pytest* cache directory.
    """

    message_box("Cleaning project...")

    patterns = ["build", "*.egg-info", "dist"]

    if docs:
        patterns.append("docs/_build")
        patterns.append("docs/generated")

    if bytecode:
        patterns.append("**/__pycache__")
        patterns.append("**/*.pyc")

    if mypy:
        patterns.append(".mypy_cache")

    if pytest:
        patterns.append(".pytest_cache")

    for pattern in patterns:
        ctx.run(f"rm -rf {pattern}")


@task
def precommit(ctx: Context) -> None:
    """
    Run the "pre-commit" hooks on the codebase.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "pre-commit" hooks on the codebase...')
    ctx.run("pre-commit run --all-files")


@task
def tests(ctx: Context) -> None:
    """
    Run the unit tests with *Pytest*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "Pytest"...')
    ctx.run("pytest --doctest-modules tests")


@task
def requirements(ctx: Context) -> None:
    """
    Export the *requirements.txt* file.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Exporting "requirements.txt" file...')
    ctx.run('uv export --no-hashes --all-extras | grep -v "-e \\." > requirements.txt')


@task(clean, precommit, tests, requirements)
def build(ctx: Context) -> None:
    """
    Build the project and runs dependency tasks, i.e., *docs*, *todo*, and
    *preflight*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Building...")
    ctx.run("uv build")
    ctx.run("twine check dist/*")


@task(precommit, tests, requirements)
def docker_build(ctx: Context) -> None:
    """
    Build the *docker* image.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Building "docker" image...')

    for architecture in ("arm64", "amd64"):
        ctx.run(
            f"docker build --platform=linux/{architecture} "
            f"-t {ORG}/{CONTAINER}:latest "
            f"-t {ORG}/{CONTAINER}:latest-{architecture} "
            f"-t {ORG}/{CONTAINER}:v{app.__version__}-{architecture} ."
        )


@task
def docker_remove(ctx: Context) -> None:
    """
    Stop and remove the *docker* container.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Stopping "docker" container...')
    with contextlib.suppress(Failure):
        ctx.run(f"docker stop {CONTAINER}")

    message_box('Removing "docker" container...')
    with contextlib.suppress(Failure):
        ctx.run(f"docker rm {CONTAINER}")


@task(docker_remove, docker_build)
def docker_run(ctx: Context) -> None:
    """
    Run the *docker* container.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "docker" container...')
    ctx.run(
        "docker run -d "
        f"--name={CONTAINER} "
        f"-p 8010:8000 {ORG}/{CONTAINER}:latest-{platform.uname()[4].lower()}"
    )


@task(clean, precommit, docker_run)
def docker_push(ctx: Context) -> None:
    """
    Push the *docker* container.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Pushing "docker" container...')
    ctx.run(f"docker push --all-tags {ORG}/{CONTAINER}")
