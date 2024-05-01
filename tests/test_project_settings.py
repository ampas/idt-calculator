import os
import unittest

from idt.core import constants
from idt.framework.project_settings import IDTProjectSettings


class TestIDTProjectSettings(unittest.TestCase):
    def setUp(self):
        self.project_settings = IDTProjectSettings()

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

    def test_properties(self):
        class_props = list(self.project_settings.properties)
        self.assertEqual(len(class_props), len(constants.ProjectSettingsMetaDataConstants.ALL))

    def test_to_file(self):
        actual_file = os.path.join(self.get_test_output_folder(), "project_settings.json")
        expected_file = os.path.join(self.get_test_resources_folder(), "project_settings.json")

        self.project_settings.to_file(actual_file)
        self.assertEqual(os.path.exists(expected_file), True)

    def test_from_file(self):
        self.project_settings.camera_make = "Cannon"
        json_string = self.project_settings.to_json()
        new_settings = IDTProjectSettings.from_json(json_string)
        json_string_loaded = new_settings.to_json()
        self.assertEqual(json_string, json_string_loaded)
