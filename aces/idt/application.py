"""
IDT Generator Application
=========================

Define the *IDT* generator application class.
"""

from __future__ import annotations

import logging
import re
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from colour.hints import List

from colour.utilities import attest, optional

import aces.idt.core.common
from aces.idt.core.constants import DirectoryStructure
from aces.idt.core.transform_id import generate_idt_urn, is_valid_csc_urn
from aces.idt.framework.project_settings import IDTProjectSettings
from aces.idt.generators import GENERATORS

if typing.TYPE_CHECKING:
    from aces.idt.generators.base_generator import IDTBaseGenerator

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTGeneratorApplication",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorApplication:
    """
    Define the *IDT* generator application that handles project loading and
    saving, generator selection and execution, as well as any other analytical
    computations not related to the generators, such as calculating the data
    required for display.

    Parameters
    ----------
    generator
        Name of the *IDT* generator to use.
    project_settings
        *IDT* project settings.
    """

    def __init__(
        self,
        generator: str = "IDTGeneratorLogCamera",
        project_settings: IDTProjectSettings | None = None,
    ) -> None:
        self._project_settings = optional(project_settings, IDTProjectSettings())
        self._generator = None
        self.generator = generator

    @property
    def generator_names(self) -> List:
        """
        Getter property for the available *IDT* generator names.

        Returns
        -------
        :class:`list`
            Available *IDT* generator names.
        """

        return list(GENERATORS)

    @property
    def generator(self) -> IDTBaseGenerator:
        """
        Getter and setter property for the selected *IDT* generator type.

        Returns
        -------
        :class`IDTBaseGenerator`:
            Selected *IDT* generator type.
        """

        return self._generator

    @generator.setter
    def generator(self, value: str) -> None:
        """Setter for the **self.generator** property."""

        if value not in self.generator_names:
            exception = (
                f'"{value}" generator is invalid, must be one of '
                f'"{self.generator_names}"!'
            )

            raise ValueError(exception)

        self._generator = GENERATORS[value](self.project_settings)

    @property
    def project_settings(self) -> IDTProjectSettings:
        """
        Getter and setter property for the *IDT* project settings.

        A single instance of the *IDT* project settings exists within the class
        and its values are updated rather than replaced with the new passed
        instance.

        Returns
        -------
        :class:`IDTProjectSettings`
            *IDT* project settings.
        """

        return self._project_settings

    @project_settings.setter
    def project_settings(self, value: IDTProjectSettings) -> None:
        """Setter for the **self.project_settings** property."""

        self._project_settings.update(value)

    def _update_project_settings_from_implicit_directory_structure(
        self, root_directory: Path
    ) -> None:
        """
        Update the *IDT* project settings using the sub-directory structure under
        given root directory. The sub-directory structure should be defined as
        follows::

            data
                colour_checker
                            EV
                                image1
                                image2
                grey_card
                        image1
                        image2

        Parameters
        ----------
        root_directory:
            Root directory holding the sub-directory structure.
        """

        colour_checker_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.COLOUR_CHECKER
        )
        attest(colour_checker_directory.exists())
        for exposure_directory in colour_checker_directory.iterdir():
            if re.match(r"-?\d", exposure_directory.name):
                EV = exposure_directory.name
                self.project_settings.data[DirectoryStructure.COLOUR_CHECKER][EV] = [
                    file
                    for file in (colour_checker_directory / exposure_directory).glob(
                        "*.*"
                    )
                    if not file.name.startswith(".")
                ]

        flatfield_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.FLATFIELD
        )
        if flatfield_directory.exists():
            self.project_settings.data[DirectoryStructure.FLATFIELD] = [
                file
                for file in flatfield_directory.glob("*.*")
                if not file.name.startswith(".")
            ]

        grey_card_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.GREY_CARD
        )
        if grey_card_directory.exists():
            self.project_settings.data[DirectoryStructure.GREY_CARD] = [
                file
                for file in grey_card_directory.glob("*.*")
                if not file.name.startswith(".")
            ]

    def _verify_directory(self, root_directory: Path | str) -> None:
        """
        Verify the *IDT* archive at given root directory.

        Parameters
        ----------
        root_directory
            Root directory holding the *IDT* archive and that needs to be
            verified.
        """

        for exposure in list(
            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER].keys()
        ):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data[
                    DirectoryStructure.COLOUR_CHECKER
                ].pop(exposure)
            ]

            for image in images:
                attest(image.exists())

            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER][
                float(exposure)
            ] = images

        if self.project_settings.data.get(DirectoryStructure.FLATFIELD, []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get(
                    DirectoryStructure.FLATFIELD, []
                )
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data[DirectoryStructure.FLATFIELD] = images
        else:
            self.project_settings.data[DirectoryStructure.FLATFIELD] = []

        if self.project_settings.data.get(DirectoryStructure.GREY_CARD, []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get(
                    DirectoryStructure.GREY_CARD, []
                )
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data[DirectoryStructure.GREY_CARD] = images
        else:
            self.project_settings.data[DirectoryStructure.GREY_CARD] = []

        if self.project_settings.data.get(DirectoryStructure.BLACK, []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get(
                    DirectoryStructure.BLACK, []
                )
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data[DirectoryStructure.BLACK] = images
        else:
            self.project_settings.data[DirectoryStructure.BLACK] = []

        if self.project_settings.data.get(DirectoryStructure.WHITE, []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get(
                    DirectoryStructure.WHITE, []
                )
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data[DirectoryStructure.WHITE] = images
        else:
            self.project_settings.data[DirectoryStructure.WHITE] = []

    def _verify_file_type(self) -> None:
        """
        Verify that the *IDT* archive contains a unique file type and set the
        used file type accordingly.
        """

        file_types = set()
        for value in self.project_settings.data[
            DirectoryStructure.COLOUR_CHECKER
        ].values():
            for item in value:
                file_types.add(item.suffix)

        for item in self.project_settings.data[DirectoryStructure.GREY_CARD]:
            file_types.add(item.suffix)

        if len(file_types) > 1:
            msg = f'Multiple file types found in the project settings: "{file_types}"'
            raise ValueError(msg)

        if file_types:
            self.project_settings.file_type = next(iter(file_types))

    def extract(self, archive: str, directory: str | None = None) -> str:
        """
        Extract the *IDT* archive.

        Parameters
        ----------
        archive
            Archive to extract.
        directory
            Directory to extract the archive to.

        Returns
        -------
        :class:`str`
            Extracted directory.
        """

        directory = aces.idt.core.common.extract_archive(archive, directory)
        extracted_directories = aces.idt.core.common.list_sub_directories(directory)
        root_directory = next(iter(extracted_directories))

        json_files = list(root_directory.glob("*.json"))

        if len(json_files) > 1:
            msg = 'Multiple "JSON" files found in the root directory!'
            raise ValueError(msg)
        if len(json_files) == 1:
            json_file = next(iter(json_files))
            LOGGER.info('Found explicit "%s" "IDT" project settings file.', json_file)
            self.project_settings = IDTProjectSettings.from_file(json_file)
        else:
            LOGGER.info('Assuming implicit "IDT" specification...')
            self.project_settings.camera_model = Path(archive).stem
            self._update_project_settings_from_implicit_directory_structure(
                root_directory
            )

        self.project_settings.working_directory = root_directory

        return root_directory

    def process_archive(self, archive: str | None) -> IDTBaseGenerator:
        """
        Compute the *IDT* either using given archive *zip* file path or the
        current *IDT* project settings if not given.

        Parameters
        ----------
        archive
            Archive *zip* file path.

        Returns
        -------
        :class:`IDTBaseGenerator`
            Instantiated *IDT* generator.
        """

        if self.generator is None:
            exception = 'No "IDT" generator was set!'

            raise ValueError(exception)

        if archive is not None:
            self.project_settings.working_directory = self.extract(archive)

        # Enforcing exposure values as floating point numbers.
        for exposure in list(
            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER].keys()
        ):
            images = [
                image
                for image in self.project_settings.data[
                    DirectoryStructure.COLOUR_CHECKER
                ].pop(exposure)
            ]

            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER][
                float(exposure)
            ] = images

        return self.process()

    def process(self) -> IDTBaseGenerator:
        """
        Run the *IDT* generator application process maintaining the execution steps.

        Returns
        -------
        :class:`IDTBaseGenerator`
            Instantiated *IDT* generator. after the process has been run
        """

        self.validate_project_settings()
        self.generator.sample()
        self.generator.sort()
        self.generator.remove_clipped_samples()
        self.generator.generate_LUT()
        self.generator.filter_LUT()
        self.generator.decode()
        self.generator.optimise()

        return self.generator

    def zip(
        self, output_directory: Path | str, archive_serialised_generator: bool = False
    ) -> Path:
        """
        Create a *zip* file with the output of the *IDT* application process.

        Parameters
        ----------
        output_directory : str
            Output directory for the *zip* file.
        archive_serialised_generator : bool
            Whether to serialise and archive the *IDT* generator.

        Returns
        -------
        :class:`pathlib.Path`
            *Zip* file path.
        """

        if not self.generator:
            exception = 'No "IDT" generator was set!'

            raise ValueError(exception)

        return self.generator.zip(
            output_directory, archive_serialised_generator=archive_serialised_generator
        )

    def validate_project_settings(self) -> None:
        """Run validation checks on the project settings.

        Raises
        ------
        ValueError
            If any of the validations fail
        """

        # Check the aces_transform_id is a valid idt_urn and try and auto populate it
        if not is_valid_csc_urn(self.project_settings.aces_transform_id):
            # If the aces_transform_id is not valid, generate a new one
            new_name = generate_idt_urn(
                self.project_settings.aces_user_name,
                self.project_settings.encoding_colourspace,
                self.project_settings.encoding_transfer_function,
                1,
            )
            # Update the project settings with the new name, if this is still invalid
            # it will raise an error from the setter
            self.project_settings.aces_transform_id = new_name

        valid, errors = self.project_settings.validate()
        if not valid:
            error_string = "\n".join(errors)
            msg = f"Invalid project settings\n: {error_string}"
            raise ValueError(msg)

        # Verify the directory structure and file types
        self._verify_directory(self.project_settings.working_directory)
        self._verify_file_type()
