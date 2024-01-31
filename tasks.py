"""
Invoke - Tasks
==============
"""

import contextlib
import inspect
import platform

from colour.hints import Boolean
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
    "requirements",
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
    docs: Boolean = True,
    bytecode: Boolean = False,
    mypy: Boolean = True,
    pytest: Boolean = True,
):
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
def precommit(ctx: Context):
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
def requirements(ctx):
    """
    Export the *requirements.txt* file.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    """

    message_box('Exporting "requirements.txt" file...')
    ctx.run(
        "poetry export -f requirements.txt "
        "--without-hashes "
        "--output requirements.txt"
    )


@task(precommit, requirements)
def docker_build(ctx: Context):
    """
    Build the *docker* image.

    Parameters
    ----------
    ctx : invoke.context.Context
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
def docker_remove(ctx: Context):
    """
    Stop and remove the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    """

    message_box('Stopping "docker" container...')
    with contextlib.suppress(Failure):
        ctx.run(f"docker stop {CONTAINER}")

    message_box('Removing "docker" container...')
    with contextlib.suppress(Failure):
        ctx.run(f"docker rm {CONTAINER}")


@task(docker_remove, docker_build)
def docker_run(ctx):
    """
    Run the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    """

    message_box('Running "docker" container...')
    ctx.run(
        "docker run -d "
        f"--name={CONTAINER} "
        f"-p 8010:8000 {ORG}/{CONTAINER}:latest-{platform.uname()[4].lower()}"
    )


@task(clean, precommit, docker_run)
def docker_push(ctx: Context):
    """
    Push the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    """

    message_box('Pushing "docker" container...')
    ctx.run(f"docker push --all-tags {ORG}/{CONTAINER}")
