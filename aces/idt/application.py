"""Module holds the main application class that controls all idt generation

"""
import logging
import re
from pathlib import Path
from typing import Optional

from colour.utilities import attest

import aces.idt.core.common
from aces.idt.core.constants import DirectoryStructure
from aces.idt.framework.project_settings import IDTProjectSettings
from aces.idt.generators import GENERATORS
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
    """The main application class which handles project loading and saving, the
    generator selection and execution, as well as any other analytical computation not
    related to the generators, such as calculating data needed for display

    """

    def __init__(self):
        self._project_settings = IDTProjectSettings()
        self._generator = None

    @property
    def generator_names(self):
        """Return the names of the generators available

        Returns
        -------
        list
            A list of names for the available generators

        """
        return list(GENERATORS.keys())

    @property
    def generator(self):
        """Return the current generator

        Returns
        -------
        IDTBaseGenerator
            The current generator

        """
        return self._generator

    @generator.setter
    def generator(self, value: type[IDTBaseGenerator]):
        if value not in self.generator_names:
            raise ValueError(f"Invalid generator name: {value}")

        generator_class = GENERATORS[value]
        self._generator = generator_class(self.project_settings)

    @property
    def project_settings(self):
        """Return the current project settings

        Returns
        -------
        IDTProjectSettings
            The current project settings

        """
        return self._project_settings

    @project_settings.setter
    def project_settings(self, value: IDTProjectSettings):
        """
        Set the project settings, maintains a single instance of the project settings,
        and updates the values

        Parameters
        ----------
        value: IDTProjectSettings
            The project settings to update with

        """
        self._project_settings.update(value)

    def extract_archive(self, archive: str, directory: Optional[str] = None):
        """
        Extract the specification from the *IDT* archive.

        Parameters
        ----------
        archive : str
            Archive to extract.
        directory : str, optional
            Known directory we want to extract the archive to
        """
        directory = aces.idt.core.common.extract_archive(archive, directory)
        extracted_directories = aces.idt.core.common.list_sub_directories(directory)
        root_directory = extracted_directories[0]

        json_files = list(root_directory.glob("*.json"))
        if len(json_files) > 1:
            raise ValueError("Multiple JSON files found in the root directory.")

        elif len(json_files) == 1:
            LOGGER.info(
                'Found explicit "%s" "IDT" project settings file.', json_files[0]
            )
            self.project_settings = IDTProjectSettings.from_file(json_files[0])
        else:
            LOGGER.info('Assuming implicit "IDT" specification...')
            self.project_settings.camera_model = Path(archive).stem
            self._update_data_from_file_structure(root_directory)

        self._update_data_with_posix_paths(root_directory)
        self._validate_images()
        return directory

    def _update_data_from_file_structure(self, root_directory: Path):
        """For the root directory of the extracted archive, update the project
        settings 'data' with the file structure found on disk, when no project_settings
        file is stored on disk. Assuming the following structure:

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
        root_directory: The folder we want to parse

        """
        colour_checker_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.COLOUR_CHECKER
        )
        attest(colour_checker_directory.exists())
        for exposure_directory in colour_checker_directory.iterdir():
            if re.match(r"-?\d", exposure_directory.name):
                EV = exposure_directory.name
                self.project_settings.data[DirectoryStructure.COLOUR_CHECKER][
                    EV
                ] = list((colour_checker_directory / exposure_directory).glob("*.*"))

        flatfield_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.FLATFIELD
        )
        if flatfield_directory.exists():
            self.project_settings.data[DirectoryStructure.FLATFIELD] = list(
                flatfield_directory.glob("*.*")
            )
        grey_card_directory = (
            root_directory / DirectoryStructure.DATA / DirectoryStructure.GREY_CARD
        )
        if grey_card_directory.exists():
            self.project_settings.data[DirectoryStructure.GREY_CARD] = list(
                flatfield_directory.glob("*.*")
            )

    def _update_data_with_posix_paths(self, root_directory: Path):
        """Update the project settings 'data' with the POSIX paths vs string paths.

        Parameters
        ----------
        root_directory : Path
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

    def process_from_archive(self, archive: str):
        """Process the IDT based on the zip archive provided

        Parameters
        ----------
        archive: str
            The filepath to the archive

        Returns
        -------
        IDTBaseGenerator
            returns the generator after it is finished
        """
        if not self.generator:
            raise ValueError("No Idt Generator Set")

        self.project_settings.working_directory = self.extract_archive(archive)
        self.generator.sample()
        self.generator.sort()
        self.generator.generate_LUT()
        self.generator.filter_LUT()
        self.generator.decode()
        self.generator.optimise()
        return self.generator

    def process_from_project_settings(self):
        """Process an IDT based on the project settings stored within the application

        Returns
        -------
        IDTBaseGenerator
            returns the generator after it is finished
        """
        if not self.generator:
            raise ValueError("No Idt Generator Set")

        # Ensuring that exposure values in the specification are floating point numbers.
        for exposure in list(
            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER].keys()
        ):
            images = [  # noqa: C416
                image
                for image in self.project_settings.data[
                    DirectoryStructure.COLOUR_CHECKER
                ].pop(exposure)
            ]

            self.project_settings.data[DirectoryStructure.COLOUR_CHECKER][
                float(exposure)
            ] = images

        self.generator.sample()
        self.generator.sort()
        self.generator.generate_LUT()
        self.generator.filter_LUT()
        self.generator.decode()
        self.generator.optimise()
        return self.generator

    def _validate_images(self):
        """Validate that the images provided are all the same file extension, and store
        this in the file type property
        """
        file_types = []
        for _, value in self.project_settings.data[
            DirectoryStructure.COLOUR_CHECKER
        ].items():
            for item in value:
                if not item.exists():
                    raise ValueError(f"File does not exist: {item}")
                file_types.append(item.suffix)

        for item in self.project_settings.data[DirectoryStructure.GREY_CARD]:
            if not item.exists():
                raise ValueError(f"File does not exist: {item}")
            file_types.append(item.suffix)

        if len(set(file_types)) > 1:
            raise ValueError("Multiple file types found in the project settings")

        self.project_settings.file_type = file_types[0]

    def zip(self, output_directory: str, archive_serialised_generator: bool = False):
        """Create a zip of the results from the idt creation

        Parameters
        ----------
        output_directory : str
            Output directory for the *zip* file.
        archive_serialised_generator : bool
            Whether to serialise and archive the *IDT* generator.

        Returns
        -------
        :class:`str`
            *Zip* file path.
        """
        if not self.generator:
            raise ValueError("No Idt Generator Set")

        return self.generator.zip(
            output_directory, archive_serialised_generator=archive_serialised_generator
        )
