"""Module for unit testing the project settings

"""
import os

from test_utils import TestIDTBase

from aces.idt.core import constants
from aces.idt.framework.project_settings import IDTProjectSettings


class TestIDTProjectSettings(TestIDTBase):
    """Class which holds the unit tests for the project settings"""

    def setUp(self):
        """Set up a new project settings object"""
        self.project_settings = IDTProjectSettings()

    def test_properties(self):
        """Tests the properties on the project settings"""
        class_props = list(self.project_settings.properties)
        self.assertEqual(
            len(class_props), len(constants.ProjectSettingsMetaDataConstants.ALL)
        )

    def test_to_file(self):
        """Test serializing the project settings to file"""
        actual_file = os.path.join(
            self.get_test_output_folder(), "project_settings.json"
        )
        expected_file = os.path.join(
            self.get_test_resources_folder(), "project_settings.json"
        )

        self.project_settings.to_file(actual_file)
        self.assertEqual(os.path.exists(expected_file), True)

    def test_from_file(self):
        """Test loading the file from disk"""
        self.project_settings.camera_make = "Cannon"
        json_string = self.project_settings.to_json()
        new_settings = IDTProjectSettings.from_json(json_string)
        json_string_loaded = new_settings.to_json()
        self.assertEqual(json_string, json_string_loaded)
