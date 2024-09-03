"""
IDT Generator Application
=========================

Define the *IDT* generator application class.
"""

import logging
import re
from pathlib import Path

from colour.hints import List
from colour.utilities import attest, optional

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
        generator: str = "IDTGeneratorProsumerCamera",
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
    def generator(self, value: str):
        """Setter for the **self.generator** property."""

        if value not in self.generator_names:
            raise ValueError(
                f'"{value}" generator is invalid, must be one of '
                f'"{self.generator_names}"!'
            )

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
    def project_settings(self, value: IDTProjectSettings):
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

    def _verify_archive(self, root_directory: Path | str) -> None:
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

    def _verify_file_type(self) -> None:
        """
        Verify that the *IDT* archive contains a unique file type and set the
        used file type accordingly.
        """

        file_types = set()
        for _, value in self.project_settings.data[
            DirectoryStructure.COLOUR_CHECKER
        ].items():
            for item in value:
                file_types.add(item.suffix)

        for item in self.project_settings.data[DirectoryStructure.GREY_CARD]:
            file_types.add(item.suffix)

        if len(file_types) > 1:
            raise ValueError(
                f'Multiple file types found in the project settings: "{file_types}"'
            )

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
            raise ValueError('Multiple "JSON" files found in the root directory!')
        elif len(json_files) == 1:
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

        self._verify_archive(root_directory)
        self._verify_file_type()

        return directory

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
            raise ValueError('No "IDT" generator was selected!')

        if archive is not None:
            self.project_settings.working_directory = self.extract(archive)

        # Enforcing exposure values as floating point numbers.
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

        return self.process()

    def process(self) -> IDTBaseGenerator:
        """Run the *IDT* generator application process maintaining the execution steps

        Returns
        -------
        :class:`IDTBaseGenerator`
            Instantiated *IDT* generator. after the process has been run

        """
        self.generator.sample()
        self.generator.sort()
        self.generator.remove_clipping()
        self.generator.generate_LUT()
        self.generator.filter_LUT()
        self.generator.decode()
        self.generator.optimise()
        return self.generator

    def zip(
        self, output_directory: Path | str, archive_serialised_generator: bool = False
    ) -> str:
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
        :class:`str`
            *Zip* file path.
        """

        if not self.generator:
            raise ValueError("No Idt Generator Set")

        return self.generator.zip(
            output_directory, archive_serialised_generator=archive_serialised_generator
        )
