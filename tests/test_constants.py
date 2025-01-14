"""
Unit tests for the constants module
"""
from aces.idt.core import constants
from tests.test_utils import TestIDTBase


class Test_IDTConstants(TestIDTBase):
    """Class which holds the unit tests for the constants module"""

    def test_class_property_deprecation(self):
        """Test the deprecation of the classmethod and  property in python3.11"""
        self.assertEqual(len(constants.CAT.ALL), 13)
