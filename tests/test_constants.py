"""Define the unit tests for the :mod:`aces.idt.core.constants` module."""

from __future__ import annotations

from aces.idt.core.constants import CAT
from tests.test_utils import TestIDTBase

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "TestCAT",
]


class TestCAT(TestIDTBase):
    """
    Define the unit tests for the :class:`aces.idt.CAT` class.
    """

    def test_ALL(self) -> None:
        """
        Test the :class:`aces.idt.CAT.ALL` property.

        A previous implementation of :class:`aces.idt.CAT.ALL` property was
        combining classmethod and property decorators which was deprecated
        in Python 3.11.
        """

        self.assertEqual(len(CAT.ALL), 13)
