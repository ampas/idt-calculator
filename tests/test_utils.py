"""Module for generic unit testing helpers

"""
import os
import unittest


class TestIDTBase(unittest.TestCase):
    """A Base class for further unit testing classes"""

    @classmethod
    def get_unit_test_folder(cls):
        """Get the folder the unit test file is stored within

        Returns
        -------
        str:
            The folder path holding the file which is running
        """
        script_file_path = os.path.abspath(__file__)
        return os.path.dirname(script_file_path)

    @classmethod
    def get_test_output_folder(cls):
        """Get the unit test output folder

        Returns
        -------
        str
            The unit test output folder
        """
        return os.path.join(cls.get_unit_test_folder(), "output")

    @classmethod
    def get_test_resources_folder(cls):
        """Get the unit test resource folder

        Returns
        -------
        str
            The unit test resource folder
        """
        return os.path.join(cls.get_unit_test_folder(), "resources")
