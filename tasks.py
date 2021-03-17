# -*- coding: utf-8 -*-
"""
Invoke - Tasks
==============
"""

from invoke import task
from invoke.exceptions import Failure

from colour.utilities import message_box

import app

__author__ = 'Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw'
__copyright__ = ('Copyright (C) 2020-2021 '
                 'Academy of Motion Picture Arts and Sciences')
__license__ = 'Academy of Motion Picture Arts and Sciences License Terms'
__maintainer__ = 'Academy of Motion Picture Arts and Sciences'
__email__ = 'acessupport@oscars.org'
__status__ = 'Production'

__all__ = [
    'APPLICATION_NAME', 'ORG', 'CONTAINER', 'clean', 'quality', 'formatting',
    'requirements', 'docker_build', 'docker_remove', 'docker_run'
]

APPLICATION_NAME = app.__application_name__

ORG = 'ampas'

CONTAINER = APPLICATION_NAME.replace(' ', '').lower()


@task
def clean(ctx, bytecode=False):
    """
    Cleans the project.

    Parameters
    ----------
    bytecode : bool, optional
        Whether to clean the bytecode files, e.g. *.pyc* files.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Cleaning project...')

    patterns = []

    if bytecode:
        patterns.append('**/*.pyc')
        patterns.append('**/__pycache__')

    for pattern in patterns:
        ctx.run("rm -rf {}".format(pattern))


@task
def quality(ctx, flake8=True):
    """
    Checks the codebase with *Flake8*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    flake8 : bool, optional
        Whether to check the codebase with *Flake8*.

    Returns
    -------
    bool
        Task success.
    """

    if flake8:
        message_box('Checking codebase with "Flake8"...')
        ctx.run('flake8')


@task
def formatting(ctx, yapf=True):
    """
    Formats the codebase with *Yapf*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    yapf : bool, optional
        Whether to format the codebase with *Yapf*.

    Returns
    -------
    bool
        Task success.
    """

    if yapf:
        message_box('Formatting codebase with "Yapf"...')
        ctx.run('yapf -p -i -r .')


@task
def requirements(ctx):
    """
    Exports the *requirements.txt* file.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Exporting "requirements.txt" file...')
    ctx.run('poetry run pip list --format=freeze | '
            'egrep -v "colour==" | '
            'sed -e "s|colour-science==.*|'
            'git+git://github.com/colour-science/colour@develop#'
            'egg=colour|g" '
            '> requirements.txt')


@task(requirements)
def docker_build(ctx):
    """
    Builds the *docker* image.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Building "docker" image...')

    ctx.run('docker build -t {0}/{1}:latest -t {0}/{1}:v{2} .'.format(
        ORG, CONTAINER, app.__version__))


@task
def docker_remove(ctx):
    """
    Stops and remove the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Stopping "docker" container...')
    try:
        ctx.run('docker stop {0}'.format(CONTAINER))
    except Failure:
        pass

    message_box('Removing "docker" container...')
    try:
        ctx.run('docker rm {0}'.format(CONTAINER))
    except Failure:
        pass


@task(docker_remove, docker_build)
def docker_run(ctx):
    """
    Runs the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Running "docker" container...')
    ctx.run('docker run -d '
            '--name={1} '
            '-p 8010:8000 {0}/{1}'.format(ORG, CONTAINER))


@task
def docker_push(ctx):
    """
    Pushes the *docker* container.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Pushing "docker" container...')
    ctx.run('docker push {0}/{1}'.format(ORG, CONTAINER))
