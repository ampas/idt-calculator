"""
Invoke - Tasks
==============
"""

from invoke import Context, task
from invoke.exceptions import Failure

from colour.hints import Boolean
from colour.utilities import message_box

import app

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


def _patch_invoke_annotations_support():
    """See https://github.com/pyinvoke/invoke/issues/357."""

    import invoke
    from unittest.mock import patch
    from inspect import getfullargspec, ArgSpec

    def patched_inspect_getargspec(function):
        spec = getfullargspec(function)
        return ArgSpec(*spec[0:4])

    org_task_argspec = invoke.tasks.Task.argspec

    def patched_task_argspec(*args, **kwargs):
        with patch(
            target="inspect.getargspec", new=patched_inspect_getargspec
        ):
            return org_task_argspec(*args, **kwargs)

    invoke.tasks.Task.argspec = patched_task_argspec


_patch_invoke_annotations_support()


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
        "poetry run pip list --format=freeze | "
        'egrep -v "idt-calculator=" '
        "> requirements.txt"
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

    ctx.run(
        "docker build -t {0}/{1}:latest -t {0}/{1}:v{2} .".format(
            ORG, CONTAINER, app.__version__
        )
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
    try:
        ctx.run(f"docker stop {CONTAINER}")
    except Failure:
        pass

    message_box('Removing "docker" container...')
    try:
        ctx.run(f"docker rm {CONTAINER}")
    except Failure:
        pass


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
        "--name={1} "
        "-p 8010:8000 {0}/{1}".format(ORG, CONTAINER)
    )


@task
def docker_push(ctx: Context):
    """
    Push the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    """

    message_box('Pushing "docker" container...')
    ctx.run(f"docker push {ORG}/{CONTAINER}")
