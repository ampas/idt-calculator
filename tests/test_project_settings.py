"""
Define the unit tests for the :class:`aces.idt.IDTProjectSettings`class.
"""

from __future__ import annotations

import json
import os

from aces.idt.core import constants
from aces.idt.framework.project_settings import IDTProjectSettings
from tests.test_utils import TestIDTBase

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "TestIDTProjectSettings",
]


class TestIDTProjectSettings(TestIDTBase):
    """
    Define the unit tests for the :class:`aces.idt.IDTProjectSettings`
    class.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        self.project_settings = IDTProjectSettings()

    def test_properties(self) -> None:
        """Test :class:`aces.idt.IDTProjectSettings` class properties."""

        class_props = list(self.project_settings.properties)
        self.assertEqual(
            len(class_props), len(constants.ProjectSettingsMetadataConstants.ALL)
        )

    def test_to_file(self) -> None:
        """Test :class:`aces.idt.IDTProjectSettings.to_file` method."""

        actual_file = os.path.join(
            self.get_test_output_folder(), "project_settings.json"
        )
        expected_file = os.path.join(
            self.get_test_resources_folder(), "project_settings.json"
        )

        self.project_settings.to_file(actual_file)
        self.assertEqual(os.path.exists(expected_file), True)

    def test_from_json(self) -> None:
        """Test :class:`aces.idt.IDTProjectSettings.from_json` method."""

        self.project_settings.camera_make = "Cannon"
        json_string = self.project_settings.to_json()
        new_settings = IDTProjectSettings.from_json(json_string)
        json_string_loaded = new_settings.to_json()
        self.assertEqual(json_string, json_string_loaded)

    def test_from_directory(self) -> None:
        """Test :class:`aces.idt.IDTProjectSettings.from_directory` method."""

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

    def test_property_names_and_metadata_names_equality(self) -> None:
        """Test that the property names are equal to the metadata names."""

        for name, prop in self.project_settings.properties:
            self.assertEqual(name, prop.metadata.name)
