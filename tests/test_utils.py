"""Module for generic unit testing helpers"""

from __future__ import annotations

import os
import unittest
from typing import TypeVar

T = TypeVar("T", bound="TestIDTBase")

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "TestIDTBase",
]


class TestIDTBase(unittest.TestCase):
    """Define a base class for other unit testing classes."""

    @classmethod
    def get_unit_test_folder(cls: type[T]) -> T:
        """
        Get the folder the unit test file is stored within.

        Returns
        -------
        :class:`str`
            The folder path holding the file which is running.
        """

        script_file_path = os.path.abspath(__file__)

        return os.path.dirname(script_file_path)

    @classmethod
    def get_test_output_folder(cls: type[T]) -> T:
        """
        Get the unit test output folder.

        Returns
        -------
        :class:`str`
            The unit test output folder.
        """

        return os.path.join(cls.get_unit_test_folder(), "output")

    @classmethod
    def get_test_resources_folder(cls: type[T]) -> T:
        """Get the unit test resource folder.

        Returns
        -------
        :class:`str`
            The unit test resource folder
        """

        return os.path.join(cls.get_unit_test_folder(), "resources")
