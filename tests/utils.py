import os
import unittest


class TestIDTBase(unittest.TestCase):

    @classmethod
    def get_unit_test_folder(cls):
        script_file_path = os.path.abspath(__file__)
        return os.path.dirname(script_file_path)

    @classmethod
    def get_test_output_folder(cls):
        return os.path.join(cls.get_unit_test_folder(), "output")

    @classmethod
    def get_test_resources_folder(cls):
        return os.path.join(cls.get_unit_test_folder(), "resources")