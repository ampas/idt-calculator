"""Module for unit testing the project settings

"""
import json
import os

from aces.idt.core import constants
from aces.idt.framework.project_settings import IDTProjectSettings
from tests.test_utils import TestIDTBase


class TestIDTProjectSettings(TestIDTBase):
    """Class which holds the unit tests for the project settings"""

    def setUp(self):
        """Set up a new project settings object"""
        self.project_settings = IDTProjectSettings()

    def test_properties(self):
        """Tests the properties on the project settings"""
        class_props = list(self.project_settings.properties)
        self.assertEqual(
            len(class_props), len(constants.ProjectSettingsMetadataConstants.ALL)
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

    def test_from_directory(self):
        """Test creating new project from directory"""
        expected_file = os.path.join(
            self.get_test_resources_folder(), "example_from_folder.json"
        )

        actual_file = os.path.join(
            self.get_test_resources_folder(), "synthetic_001", "test_project.json"
        )

        folder_path = os.path.join(self.get_test_resources_folder(), "synthetic_001")
        new_settings = IDTProjectSettings.from_directory(folder_path)
        new_settings.to_file(actual_file)

        with open(actual_file) as actual_handle:
            actual = json.load(actual_handle)

        with open(expected_file) as expected_handle:
            expected = json.load(expected_handle)

        self.maxDiff = None

        self.assertEqual(actual, expected)

    def test_property_vs_metadata_name(self):
        """Test that the property name matches the metadata name"""
        for name, prop in self.project_settings.properties:
            self.assertEqual(name, prop.metadata.name)
