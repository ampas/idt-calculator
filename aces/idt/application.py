"""Module holds the main application class that controls all idt generation

"""
import logging
import re
from pathlib import Path
from typing import Optional

from colour.utilities import attest

from aces.idt.core import utilities
from aces.idt.core.common import RGB_COLORCHECKER_CLASSIC_ACES
from aces.idt.core.constants import DataFolderStructure
from aces.idt.framework.project_settings import IDTProjectSettings
from aces.idt.generators import ALL_GENERATORS

logger = logging.getLogger(__name__)


class IDTGeneratorApplication:
    """The main application class which handles project loading and saving, the
    generator selection and execution, as well as any other analytical computation not
    related to the generators, such as calculating data needed for display

    """

    def __init__(self, working_directory=None, cleanup=True):
        self._project_settings = IDTProjectSettings()

        self._all_generators = ALL_GENERATORS
        self._idt_generator = None
        self._file_type = None
        self._working_directory = working_directory
        self._cleanup = cleanup
        self._reference_colour_checker = RGB_COLORCHECKER_CLASSIC_ACES
        self._sigma = 16

    @property
    def all_generators(self):
        """Return the names of the generators available

        Returns
        -------
        list
            A list of names for the available generators

        """
        return list(self._all_generators.keys())

    @property
    def idt_generator(self):
        """Return the current generator

        Returns
        -------
        IDTBaseGenerator
            The current generator

        """
        return self._idt_generator

    @idt_generator.setter
    def idt_generator(self, value):
        if value not in self.all_generators:
            raise ValueError(f"Invalid generator name: {value}")

        generator_class = self._all_generators.get(value)
        self._idt_generator = generator_class(self)

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
        if not isinstance(value, IDTProjectSettings):
            raise TypeError("project_settings must be of type IDTProjectSettings")
        self._project_settings = value

    @property
    def file_type(self):
        """Return the current file type of the files which are being processed

        Returns
        -------
        str
            The file type of the files within the archive or project
        """
        return self._file_type

    @file_type.setter
    def file_type(self, value):
        self._file_type = value

    @property
    def working_directory(self):
        """Return the working directory

        Returns
        -------
        PosixPath
            The current working directory

        """
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value):
        self._working_directory = value

    @property
    def cleanup(self):
        """Returns whether we should perform a cleanup or not

        Returns
        -------
        bool
            Cleanup the directory or not
        """
        return self._cleanup

    @cleanup.setter
    def cleanup(self, value):
        self._cleanup = value

    # TODO: The following properties are not currently part of the schema,
    #  ou should be able to serialize a project
    #  and run it without need to set anything else up as a headless command
    @property
    def reference_colour_checker(self):
        """Return the reference colour checker values

        Returns
        -------
        Array
            Reference colour checker values
        """
        return self._reference_colour_checker

    @reference_colour_checker.setter
    def reference_colour_checker(self, value):
        self._reference_colour_checker = value

    @property
    def sigma(self):
        """Return the sigma value

        Returns
        -------
        int
            The sigma value
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def optimisation_factory(self):
        """Return the optimisation factory we want to use

        Returns
        -------
        str
            The name of the optimisation factory
        """
        return self._optimisation_factory

    @optimisation_factory.setter
    def optimisation_factory(self, value):
        self._optimisation_factory = value

    def extract_archive(self, archive: str, directory: Optional[str] = None):
        """
        Extract the specification from the *IDT* archive.
        """
        directory = utilities.extract_archive(archive, directory)
        extracted_directories = utilities.list_sub_directories(directory)
        root_directory = extracted_directories[0]

        json_files = list(root_directory.glob("*.json"))
        if len(json_files) > 1:
            raise ValueError("Multiple JSON files found in the root directory.")

        elif len(json_files) == 1:
            logger.info(
                'Found explicit "%s" "IDT" project settings file.', json_files[0]
            )
            self.project_settings = IDTProjectSettings.from_file(json_files[0])
        else:
            logger.info('Assuming implicit "IDT" specification...')
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
            root_directory
            / DataFolderStructure.DATA
            / DataFolderStructure.COLOUR_CHECKER
        )
        attest(colour_checker_directory.exists())
        for exposure_directory in colour_checker_directory.iterdir():
            if re.match(r"-?\d", exposure_directory.name):
                EV = exposure_directory.name
                self.project_settings.data[DataFolderStructure.COLOUR_CHECKER][
                    EV
                ] = list((colour_checker_directory / exposure_directory).glob("*.*"))

        # TODO Is flatfield used? ive only ever scene colour checker and grey_card used
        flatfield_directory = root_directory / DataFolderStructure.DATA / "flatfield"
        if flatfield_directory.exists():
            self.project_settings.data["flatfield"] = list(
                flatfield_directory.glob("*.*")
            )
        grey_card_directory = (
            root_directory / DataFolderStructure.DATA / DataFolderStructure.GREY_CARD
        )
        if grey_card_directory.exists():
            self.project_settings.data[DataFolderStructure.GREY_CARD] = list(
                flatfield_directory.glob("*.*")
            )

    def _update_data_with_posix_paths(self, root_directory: Path):
        """Update the project settings 'data' with the POSIX paths vs string paths.

        Parameters
        ----------
        root_directory : Path
        """
        for exposure in list(
            self.project_settings.data[DataFolderStructure.COLOUR_CHECKER].keys()
        ):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data[
                    DataFolderStructure.COLOUR_CHECKER
                ].pop(exposure)
            ]

            for image in images:
                attest(image.exists())

            self.project_settings.data[DataFolderStructure.COLOUR_CHECKER][
                float(exposure)
            ] = images

        # TODO Is flatfield used? ive only ever scene colour checker and grey_card used
        if self.project_settings.data.get("flatfield", []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get("flatfield", [])
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data["flatfield"] = images
        else:
            self.project_settings.data["flatfield"] = []

        if self.project_settings.data.get(DataFolderStructure.GREY_CARD, []):
            images = [
                Path(root_directory) / image
                for image in self.project_settings.data.get(
                    DataFolderStructure.GREY_CARD, []
                )
            ]
            for image in images:
                attest(image.exists())

            self.project_settings.data[DataFolderStructure.GREY_CARD] = images
        else:
            self.project_settings.data[DataFolderStructure.GREY_CARD] = []

    def process_from_archive(self, archive):
        """Process the IDT based on the zip archive provided

        Parameters
        ----------
        archive: str The filepath to the archive

        Returns
        -------
        IDTBaseGenerator
            returns the generator after it is finished
        """
        if not self.idt_generator:
            raise ValueError("No Idt Generator Set")

        self.working_directory = self.extract_archive(archive, self.working_directory)
        self.idt_generator.sample()
        self.idt_generator.sort()
        self.idt_generator.generate_LUT()
        self.idt_generator.filter_LUT()
        self.idt_generator.decode()
        self.idt_generator.optimise()
        return self.idt_generator

    def process_from_project_settings(self):
        """Process an IDT based on the project settings stored within the application

        Returns
        -------
        IDTBaseGenerator
            returns the generator after it is finished
        """
        if not self.idt_generator:
            raise ValueError("No Idt Generator Set")

        # Ensuring that exposure values in the specification are floating point numbers.
        for exposure in list(
            self.project_settings.data[DataFolderStructure.COLOUR_CHECKER].keys()
        ):
            images = [  # noqa: C416
                image
                for image in self.project_settings.data[
                    DataFolderStructure.COLOUR_CHECKER
                ].pop(exposure)
            ]

            self.project_settings.data[DataFolderStructure.COLOUR_CHECKER][
                float(exposure)
            ] = images

        self.idt_generator.sample()
        self.idt_generator.sort()
        self.idt_generator.generate_LUT()
        self.idt_generator.filter_LUT()
        self.idt_generator.decode()
        self.idt_generator.optimise()
        return self.idt_generator

    def _validate_images(self):
        """Validate that the images provided are all the same file extension, and store
        this in the file type property
        """
        file_types = []
        for _, value in self.project_settings.data[
            DataFolderStructure.COLOUR_CHECKER
        ].items():
            for item in value:
                if not item.exists():
                    raise ValueError(f"File does not exist: {item}")
                file_types.append(item.suffix)

        for item in self.project_settings.data[DataFolderStructure.GREY_CARD]:
            if not item.exists():
                raise ValueError(f"File does not exist: {item}")
            file_types.append(item.suffix)

        if len(set(file_types)) > 1:
            raise ValueError("Multiple file types found in the project settings")

        self.file_type = file_types[0]

    def zip(self, output_directory, archive_serialised_generator=False):
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
        if not self.idt_generator:
            raise ValueError("No Idt Generator Set")

        return self.idt_generator.zip(
            output_directory, archive_serialised_generator=archive_serialised_generator
        )
