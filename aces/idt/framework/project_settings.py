"""Module which contains the ProjectSettings for the IDT maker acts as the data model,
serialization, loading and saving of a project.

"""
from __future__ import annotations

import os
from collections import OrderedDict

from colour.utilities import multiline_str

from aces.idt.core import common
from aces.idt.core.common import OPTIMISATION_FACTORIES
from aces.idt.core.constants import DataFolderStructure
from aces.idt.core.constants import ProjectSettingsMetaDataConstants as PsMdC
from aces.idt.core.structures import BaseSerializable, idt_metadata_property
from aces.idt.core.utilities import format_exposure_key, sort_exposure_keys


class IDTProjectSettings(BaseSerializable):
    """Class which holds the project settings for the IDT maker. The order the
    properties are created is the order they are serialized and deserialized.
    """

    def __init__(self):
        super().__init__()
        self._schema_version = IDTProjectSettings.schema_version.metadata.default_value
        self._aces_transform_id = (
            IDTProjectSettings.aces_transform_id.metadata.default_value
        )
        self._aces_user_name = IDTProjectSettings.aces_user_name.metadata.default_value
        self._camera_make = IDTProjectSettings.camera_make.metadata.default_value
        self._camera_model = IDTProjectSettings.camera_model.metadata.default_value
        self._iso = IDTProjectSettings.iso.metadata.default_value
        self._temperature = IDTProjectSettings.temperature.metadata.default_value
        self._additional_camera_settings = (
            IDTProjectSettings.additional_camera_settings.metadata.default_value
        )
        self._lighting_setup_description = (
            IDTProjectSettings.lighting_setup_description.metadata.default_value
        )
        self._debayering_platform = (
            IDTProjectSettings.debayering_platform.metadata.default_value
        )
        self._debayering_settings = (
            IDTProjectSettings.debayering_settings.metadata.default_value
        )
        self._encoding_colourspace = (
            IDTProjectSettings.encoding_colourspace.metadata.default_value
        )
        self._rgb_display_colourspace = (
            IDTProjectSettings.rgb_display_colourspace.metadata.default_value
        )
        self._cat = IDTProjectSettings.cat.metadata.default_value
        self._optimisation_space = (
            IDTProjectSettings.optimisation_space.metadata.default_value
        )
        self._illuminant_interpolator = (
            IDTProjectSettings.illuminant_interpolator.metadata.default_value
        )
        self._decoding_method = (
            IDTProjectSettings.decoding_method.metadata.default_value
        )
        self._ev_range = IDTProjectSettings.ev_range.metadata.default_value
        self._grey_card_reference = (
            IDTProjectSettings.grey_card_reference.metadata.default_value
        )
        self._lut_size = IDTProjectSettings.lut_size.metadata.default_value
        self._lut_smoothing = IDTProjectSettings.lut_smoothing.metadata.default_value
        self._data = IDTProjectSettings.data.metadata.default_value
        self._working_directory = (
            IDTProjectSettings.working_directory.metadata.default_value
        )
        self._cleanup = IDTProjectSettings.cleanup.metadata.default_value
        self._reference_colour_checker = (
            IDTProjectSettings.reference_colour_checker.metadata.default_value
        )
        self._illuminant = IDTProjectSettings.illuminant.metadata.default_value
        self._sigma = IDTProjectSettings.sigma.metadata.default_value
        self._file_type = IDTProjectSettings.file_type.metadata.default_value
        self._ev_weights = IDTProjectSettings.ev_weights.metadata.default_value
        self._optimization_kwargs = (
            IDTProjectSettings.optimization_kwargs.metadata.default_value
        )

    @idt_metadata_property(metadata=PsMdC.SCHEMA_VERSION)
    def schema_version(self):
        """Return the schema version

        Returns
        -------
        str
            The project settings schema version

        """
        return self._schema_version

    @idt_metadata_property(metadata=PsMdC.ACES_TRANSFORM_ID)
    def aces_transform_id(self):
        """Return the aces transform id

        Returns
        -------
        str
            The ACES transform ID

        """
        return self._aces_transform_id

    @idt_metadata_property(metadata=PsMdC.ACES_USER_NAME)
    def aces_user_name(self):
        """Return the aces user name

        Returns
        -------
        str
            The ACES user name

        """
        return self._aces_user_name

    @idt_metadata_property(metadata=PsMdC.CAMERA_MAKE)
    def camera_make(self):
        """Return the camera make

        Returns
        -------
        str
            The camera make

        """
        return self._camera_make

    @idt_metadata_property(metadata=PsMdC.CAMERA_MODEL)
    def camera_model(self):
        """Return the camera model

        Returns
        -------
        str
            The camera model

        """
        return self._camera_model

    @idt_metadata_property(metadata=PsMdC.ISO)
    def iso(self):
        """Return the ISO

        Returns
        -------
        int
            The ISO

        """
        return self._iso

    @idt_metadata_property(metadata=PsMdC.TEMPERATURE)
    def temperature(self):
        """Return the colour temperature in kelvin

        Returns
        -------
        int
            The temperature

        """
        return self._temperature

    @idt_metadata_property(metadata=PsMdC.ADDITIONAL_CAMERA_SETTINGS)
    def additional_camera_settings(self):
        """Return the additional camera settings

        Returns
        -------
        str
            The additional camera settings

        """
        return self._additional_camera_settings

    @idt_metadata_property(metadata=PsMdC.LIGHTING_SETUP_DESCRIPTION)
    def lighting_setup_description(self):
        """Return the lighting setup description

        Returns
        -------
        str
            The lighting setup description

        """
        return self._lighting_setup_description

    @idt_metadata_property(metadata=PsMdC.DEBAYERING_PLATFORM)
    def debayering_platform(self):
        """Return the debayering platform

        Returns
        -------
        str
            The debayering platform

        """
        return self._debayering_platform

    @idt_metadata_property(metadata=PsMdC.DEBAYERING_SETTINGS)
    def debayering_settings(self):
        """Return the debayering settings settings

        Returns
        -------
        str
            The debayering settings

        """
        return self._debayering_settings

    @idt_metadata_property(metadata=PsMdC.ENCODING_COLOUR_SPACE)
    def encoding_colourspace(self):
        """Return the encoding colour space

        Returns
        -------
        str
            The encoding colour space

        """
        return self._encoding_colourspace

    @idt_metadata_property(metadata=PsMdC.RGB_DISPLAY_COLOURSPACE)
    def rgb_display_colourspace(self):
        """Return the rgb display colour space

        Returns
        -------
        str
            The RGB display colourspace

        """
        return self._rgb_display_colourspace

    @idt_metadata_property(metadata=PsMdC.CAT)
    def cat(self):
        """Return the name of the CAT

        Returns
        -------
        str
            The CAT

        """
        return self._cat

    @idt_metadata_property(metadata=PsMdC.OPTIMISATION_SPACE)
    def optimisation_space(self):
        """Return the optimisation space

        Returns
        -------
        str
            The optimisation space

        """
        return self._optimisation_space

    @idt_metadata_property(metadata=PsMdC.ILLUMINANT_INTERPOLATOR)
    def illuminant_interpolator(self):
        """Return the illuminant interpolator

        Returns
        -------
        str
            The illuminant interpolator

        """
        return self._illuminant_interpolator

    @idt_metadata_property(metadata=PsMdC.DECODING_METHOD)
    def decoding_method(self):
        """Return the decoding method

        Returns
        -------
        str
            The decoding method

        """
        return self._decoding_method

    @idt_metadata_property(metadata=PsMdC.EV_RANGE)
    def ev_range(self):
        """Return the ev range

        Returns
        -------
        list
            The EV range

        """
        return self._ev_range

    @idt_metadata_property(metadata=PsMdC.GREY_CARD_REFERENCE)
    def grey_card_reference(self):
        """Return the grey card reference values

        Returns
        -------
        list
            The grey card reference

        """
        return self._grey_card_reference

    @idt_metadata_property(metadata=PsMdC.LUT_SIZE)
    def lut_size(self):
        """Return the lut size

        Returns
        -------
        int
            The size of the lut

        """
        return self._lut_size

    @idt_metadata_property(metadata=PsMdC.LUT_SMOOTHING)
    def lut_smoothing(self):
        """Return the lut smoothing

        Returns
        -------
        int
            The smoothing amount for the lut

        """
        return self._lut_smoothing

    @idt_metadata_property(metadata=PsMdC.DATA)
    def data(self):
        """Return the data structure to hold the image sequences

        Returns
        -------
        dict
            The data structure to hold the image sequences

        """
        return self._data

    @idt_metadata_property(metadata=PsMdC.WORKING_DIR)
    def working_directory(self):
        """Return the working directory for the project

        Returns
        -------
        str
            The working directory for the project

        """
        return self._working_directory

    @idt_metadata_property(metadata=PsMdC.CLEAN_UP)
    def cleanup(self):
        """Return whether we want to cleanup the working dir or not

        Returns
        -------
        bool
            Whether we are cleaning up the working dir or not

        """
        return self._cleanup

    @idt_metadata_property(metadata=PsMdC.REFERENCE_COLOUR_CHECKER)
    def reference_colour_checker(self):
        """Return the reference_colour_checker

        Returns
        -------
        NDArray
            The reference colour checker
        """
        return self._reference_colour_checker

    @idt_metadata_property(metadata=PsMdC.ILLUMINANT)
    def illuminant(self):
        """Return the illuminant

        Returns
        -------
        NDArray
            The reference illuminant
        """
        return self._illuminant

    @idt_metadata_property(metadata=PsMdC.SIGMA)
    def sigma(self):
        """Return the sigma

        Returns
        -------
        int
            The sigma
        """
        return self._sigma

    @idt_metadata_property(metadata=PsMdC.FILE_TYPE)
    def file_type(self):
        """Return the file_type

        Returns
        -------
        int
            The file_type
        """
        return self._file_type

    @idt_metadata_property(metadata=PsMdC.EV_WEIGHTS)
    def ev_weights(self):
        """Return the ev weights

        Returns
        -------
        np.Array
            The ev weights to use
        """
        return self._ev_weights

    @idt_metadata_property(metadata=PsMdC.OPTIMIZATION_KWARGS)
    def optimization_kwargs(self):
        """Return the optimization_kwargs

        Returns
        -------
        dict
            The optimization_kwargs used
        """
        return self._optimization_kwargs

    def get_reference_colour_checker_samples(self):
        """Return the reference_colour_checker samples

        Returns
        -------
        NDArray
            The reference colour checker samples
        """
        return common.generate_reference_colour_checker(
            common.get_sds_colour_checker(self.reference_colour_checker),
            common.get_sds_illuminant(self.illuminant),
        )

    def get_optimization_factory(self):
        """Return the optimisation factory based on the optimisation space

        Returns
        -------
        The optimisation factory

        """
        result = OPTIMISATION_FACTORIES.get(self.optimisation_space, None)
        if result is None:
            raise ValueError(f"Optimisation space {self.optimisation_space} not found")
        return result

    def update(self, value: IDTProjectSettings):
        """Update the project settings with the given value from another project
        settings

        Parameters
        ----------
        value: IDTProjectSettings
            The project settings to update with

        """
        if not isinstance(value, IDTProjectSettings):
            raise TypeError(f"Expected IDTProjectSettings, got {type(value)} instead")

        for name, prop in value.properties:
            value2 = prop.getter(value)
            setattr(self, name, value2)

    @classmethod
    def from_folder(cls, project_name, folder_path):
        """Create a new project settings for a given folder on disk and build the data
        structure based on the files on disk

        Parameters
        ----------
        project_name: str
            The name of the project
        folder_path : str
            The folder path to the project root containing the image sequence folders
        """

        instance = cls()
        data = {
            DataFolderStructure.COLOUR_CHECKER: {},
            DataFolderStructure.GREY_CARD: [],
        }

        # Validate folder paths for colour_checker and grey_card exist
        colour_checker_path = os.path.join(
            folder_path, DataFolderStructure.DATA, DataFolderStructure.COLOUR_CHECKER
        )
        grey_card_path = os.path.join(
            folder_path, DataFolderStructure.DATA, DataFolderStructure.GREY_CARD
        )

        if not os.path.exists(colour_checker_path) or not os.path.exists(
            grey_card_path
        ):
            raise ValueError(
                "Required 'colour_checker' or 'grey_card' folder does not exist."
            )

        # Populate colour_checker data
        for root, _, files in os.walk(colour_checker_path):
            for file in files:
                exposure_value = os.path.basename(
                    root
                )  # Assuming folder names are the exposure values
                if os.path.basename(file).startswith("."):
                    pass

                if exposure_value not in data[DataFolderStructure.COLOUR_CHECKER]:
                    data[DataFolderStructure.COLOUR_CHECKER][exposure_value] = []
                absolute_file_path = os.path.join(root, file)
                data[DataFolderStructure.COLOUR_CHECKER][exposure_value].append(
                    os.path.relpath(absolute_file_path, start=folder_path)
                )

        sorted_keys = sorted(
            data[DataFolderStructure.COLOUR_CHECKER], key=sort_exposure_keys
        )

        # Create a new OrderedDict with the sorted keys
        sorted_colour_checker = OrderedDict(
            (format_exposure_key(key), data[DataFolderStructure.COLOUR_CHECKER][key])
            for key in sorted_keys
        )

        data[DataFolderStructure.COLOUR_CHECKER] = sorted_colour_checker

        # Populate grey_card data
        for root, _, files in os.walk(grey_card_path):
            files.sort()
            for file in files:
                if os.path.basename(file).startswith("."):
                    continue
                absolute_file_path = os.path.join(root, file)
                data[DataFolderStructure.GREY_CARD].append(
                    os.path.relpath(absolute_file_path, start=folder_path)
                )

        instance.data = data
        output_file = os.path.join(folder_path, f"{project_name}.json")
        instance.to_file(output_file)
        return output_file

    def __str__(self):
        """
        Return a formatted string representation of the project settings.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """
        attributes = [
            {
                "label": super().__repr__(),
                "header": True,
            },
            {"line_break": True},
        ]
        for name, _ in self.properties:
            attributes.append({"name": f"{name}", "label": f"{name}".title()})

        attributes.append(
            {"line_break": True},
        )
        return multiline_str(self, attributes)
