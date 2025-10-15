"""
Project Settings
================

Define the objects managing an *IDT* generator project settings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from colour.hints import Dict, NDArrayFloat

from colour.utilities import as_float_array, multiline_str

from aces.idt.core import (
    OPTIMISATION_FACTORIES,
    DirectoryStructure,
    MixinSerializableProperties,
)
from aces.idt.core import ProjectSettingsMetadataConstants as MetadataConstants
from aces.idt.core import (
    format_exposure_key,
    generate_reference_colour_checker,
    get_sds_colour_checker,
    get_sds_illuminant,
    metadata_property,
    sort_exposure_keys,
)
from aces.idt.core.transform_id import is_valid_csc_urn

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTProjectSettings",
]


class IDTProjectSettings(MixinSerializableProperties):
    """
    Define the project settings for an *IDT* generator.

    The properties are serialized and deserialized in the order they are
    defined in this class.

    Other Parameters
    ----------------
    kwargs
        Optional keyword arguments used to initialise the project settings.
    """

    def __init__(self, **kwargs: Dict) -> None:
        super().__init__()

        self._schema_version = IDTProjectSettings.schema_version.metadata.default_value

        self._aces_transform_id = kwargs.get(
            IDTProjectSettings.aces_transform_id.metadata.name,
            IDTProjectSettings.aces_transform_id.metadata.default_value,
        )
        self._aces_user_name = kwargs.get(
            IDTProjectSettings.aces_user_name.metadata.name,
            IDTProjectSettings.aces_user_name.metadata.default_value,
        )
        self._camera_make = kwargs.get(
            IDTProjectSettings.camera_make.metadata.name,
            IDTProjectSettings.camera_make.metadata.default_value,
        )
        self._camera_model = kwargs.get(
            IDTProjectSettings.camera_model.metadata.name,
            IDTProjectSettings.camera_model.metadata.default_value,
        )
        self._iso = kwargs.get(
            IDTProjectSettings.iso.metadata.name,
            IDTProjectSettings.iso.metadata.default_value,
        )
        self._temperature = kwargs.get(
            IDTProjectSettings.temperature.metadata.name,
            IDTProjectSettings.temperature.metadata.default_value,
        )
        self._illuminant_custom_temperature = kwargs.get(
            IDTProjectSettings.illuminant_custom_temperature.metadata.name,
            IDTProjectSettings.illuminant_custom_temperature.metadata.default_value,
        )
        self._additional_camera_settings = kwargs.get(
            IDTProjectSettings.additional_camera_settings.metadata.name,
            IDTProjectSettings.additional_camera_settings.metadata.default_value,
        )
        self._lighting_setup_description = kwargs.get(
            IDTProjectSettings.lighting_setup_description.metadata.name,
            IDTProjectSettings.lighting_setup_description.metadata.default_value,
        )
        self._debayering_platform = kwargs.get(
            IDTProjectSettings.debayering_platform.metadata.name,
            IDTProjectSettings.debayering_platform.metadata.default_value,
        )
        self._debayering_settings = kwargs.get(
            IDTProjectSettings.debayering_settings.metadata.name,
            IDTProjectSettings.debayering_settings.metadata.default_value,
        )
        self._encoding_colourspace = kwargs.get(
            IDTProjectSettings.encoding_colourspace.metadata.name,
            IDTProjectSettings.encoding_colourspace.metadata.default_value,
        )
        self._encoding_transfer_function = kwargs.get(
            IDTProjectSettings.encoding_transfer_function.metadata.name,
            IDTProjectSettings.encoding_transfer_function.metadata.default_value,
        )
        self._rgb_display_colourspace = kwargs.get(
            IDTProjectSettings.rgb_display_colourspace.metadata.name,
            IDTProjectSettings.rgb_display_colourspace.metadata.default_value,
        )
        self._cat = kwargs.get(
            IDTProjectSettings.cat.metadata.name,
            IDTProjectSettings.cat.metadata.default_value,
        )
        self._optimisation_space = kwargs.get(
            IDTProjectSettings.optimisation_space.metadata.name,
            IDTProjectSettings.optimisation_space.metadata.default_value,
        )
        self._illuminant_interpolator = kwargs.get(
            IDTProjectSettings.illuminant_interpolator.metadata.name,
            IDTProjectSettings.illuminant_interpolator.metadata.default_value,
        )
        self._decoding_method = kwargs.get(
            IDTProjectSettings.decoding_method.metadata.name,
            IDTProjectSettings.decoding_method.metadata.default_value,
        )
        self._ev_range = as_float_array(
            kwargs.get(
                IDTProjectSettings.ev_range.metadata.name,
                IDTProjectSettings.ev_range.metadata.default_value,
            )
        )
        self._grey_card_reference = as_float_array(
            kwargs.get(
                IDTProjectSettings.grey_card_reference.metadata.name,
                IDTProjectSettings.grey_card_reference.metadata.default_value,
            )
        )
        self._lut_size = kwargs.get(
            IDTProjectSettings.lut_size.metadata.name,
            IDTProjectSettings.lut_size.metadata.default_value,
        )
        self._lut_smoothing = kwargs.get(
            IDTProjectSettings.lut_smoothing.metadata.name,
            IDTProjectSettings.lut_smoothing.metadata.default_value,
        )

        self._data = IDTProjectSettings.data.metadata.default_value

        self._working_directory = kwargs.get(
            IDTProjectSettings.working_directory.metadata.name,
            IDTProjectSettings.working_directory.metadata.default_value,
        )
        self._cleanup = kwargs.get(
            IDTProjectSettings.cleanup.metadata.name,
            IDTProjectSettings.cleanup.metadata.default_value,
        )

        self._reference_colour_checker = kwargs.get(
            IDTProjectSettings.reference_colour_checker.metadata.name,
            IDTProjectSettings.reference_colour_checker.metadata.default_value,
        )
        self._illuminant = kwargs.get(
            IDTProjectSettings.illuminant.metadata.name,
            IDTProjectSettings.illuminant.metadata.default_value,
        )

        self._file_type = kwargs.get(
            IDTProjectSettings.file_type.metadata.name,
            IDTProjectSettings.file_type.metadata.default_value,
        )
        self._ev_weights = as_float_array(
            kwargs.get(
                IDTProjectSettings.ev_weights.metadata.name,
                IDTProjectSettings.ev_weights.metadata.default_value,
            )
        )
        self._optimization_kwargs = kwargs.get(
            IDTProjectSettings.optimization_kwargs.metadata.name,
            IDTProjectSettings.optimization_kwargs.metadata.default_value,
        )
        self._include_white_balance_in_clf = kwargs.get(
            IDTProjectSettings.include_white_balance_in_clf.metadata.name,
            IDTProjectSettings.include_white_balance_in_clf.metadata.default_value,
        )
        self._flatten_clf = kwargs.get(
            IDTProjectSettings.flatten_clf.metadata.name,
            IDTProjectSettings.flatten_clf.metadata.default_value,
        )
        self._include_exposure_factor_in_clf = kwargs.get(
            IDTProjectSettings.include_exposure_factor_in_clf.metadata.name,
            IDTProjectSettings.include_exposure_factor_in_clf.metadata.default_value,
        )

    @metadata_property(metadata=MetadataConstants.SCHEMA_VERSION)
    def schema_version(self) -> str:
        """
        Getter property for the schema version.

        Returns
        -------
        :class:`str`
            The project settings schema version
        """

        return self._schema_version

    @metadata_property(
        metadata=MetadataConstants.ACES_TRANSFORM_ID, validation=is_valid_csc_urn
    )
    def aces_transform_id(self) -> str:
        """
        Getter property for the *ACEStransformID*.

        Returns
        -------
        :class:`str`
            *ACEStransformID*.
        """

        return self._aces_transform_id

    @metadata_property(metadata=MetadataConstants.ACES_USER_NAME)
    def aces_user_name(self) -> str:
        """
        Getter property for the *ACESuserName*.

        Returns
        -------
        :class:`str`
           *ACESuserName*.
        """

        return self._aces_user_name

    @metadata_property(metadata=MetadataConstants.CAMERA_MAKE)
    def camera_make(self) -> str:
        """
        Getter property for the camera make.

        Returns
        -------
        :class:`str`
            Camera make.
        """

        return self._camera_make

    @metadata_property(metadata=MetadataConstants.CAMERA_MODEL)
    def camera_model(self) -> str:
        """
        Getter property for the camera model.

        Returns
        -------
        :class:`str`
            Camera model.
        """

        return self._camera_model

    @metadata_property(metadata=MetadataConstants.ISO)
    def iso(self) -> int:
        """
        Getter property for the camera ISO value.

        Returns
        -------
        :class:`int`
            Camera ISO value.
        """

        return self._iso

    @metadata_property(metadata=MetadataConstants.TEMPERATURE)
    def temperature(self) -> int:
        """
        Getter property for the camera white-balance colour temperature in
        Kelvin degrees.

        Returns
        -------
        :class:`int`
            Camera white-balance colour temperature in Kelvin degrees.
        """

        return self._temperature

    @metadata_property(metadata=MetadataConstants.ILLUMINANT_CUSTOM_TEMPERATURE)
    def illuminant_custom_temperature(self) -> int:
        """
        Getter property for the illuminant custom temperature in Kelvin degrees.

        This property is used when the illuminant is set to "Blackbody" or
        "Daylight". It specifies the correlated colour temperature (CCT) in
        Kelvin for generating the spectral distribution.

        Returns
        -------
        :class:`int`
            Illuminant custom temperature in Kelvin degrees.

        Notes
        -----
        -   For Blackbody illuminants, this temperature is used directly with
            Planck's law to generate the spectral power distribution.
        -   For Daylight illuminants, this temperature is used to generate a
            CIE D Series illuminant at the specified CCT.
        """

        return self._illuminant_custom_temperature

    @metadata_property(metadata=MetadataConstants.ADDITIONAL_CAMERA_SETTINGS)
    def additional_camera_settings(self) -> str:
        """
        Getter property for the additional camera settings.

        Returns
        -------
        :class:`str`
            Additional camera settings.
        """

        return self._additional_camera_settings

    @metadata_property(metadata=MetadataConstants.LIGHTING_SETUP_DESCRIPTION)
    def lighting_setup_description(self) -> str:
        """
        Getter property for the lighting setup description.

        Returns
        -------
        :class:`str`
            Lighting setup description.
        """

        return self._lighting_setup_description

    @metadata_property(metadata=MetadataConstants.DEBAYERING_PLATFORM)
    def debayering_platform(self) -> str:
        """
        Getter property for the debayering platform name.

        Returns
        -------
        :class:`str`
            Debayering platform name.
        """

        return self._debayering_platform

    @metadata_property(metadata=MetadataConstants.DEBAYERING_SETTINGS)
    def debayering_settings(self) -> str:
        """
        Getter property for the debayering platform settings.

        Returns
        -------
        :class:`str`
            The debayering platform settings
        """

        return self._debayering_settings

    @metadata_property(metadata=MetadataConstants.ENCODING_COLOUR_SPACE)
    def encoding_colourspace(self) -> str:
        """
        Getter property for the encoding colour space.

        Returns
        -------
        :class:`str`
            Encoding colour space.
        """

        return self._encoding_colourspace

    @metadata_property(metadata=MetadataConstants.ENCODING_TRANSFER_FUNCTION)
    def encoding_transfer_function(self) -> str:
        """
        Getter property for the encoding transfer function.

        Returns
        -------
        :class:`str`
            Encoding transfer function.
        """

        return self._encoding_transfer_function

    @metadata_property(metadata=MetadataConstants.RGB_DISPLAY_COLOURSPACE)
    def rgb_display_colourspace(self) -> str:
        """
        Getter property for the *RGB* display colour space.

        Returns
        -------
        :class:`str`
            *RGB* display colourspace.
        """

        return self._rgb_display_colourspace

    @metadata_property(metadata=MetadataConstants.CAT)
    def cat(self) -> str:
        """
        Getter property for the chromatic adaptation transform name.

        Returns
        -------
        :class:`str`
            Chromatic adaptation transform name.
        """

        return self._cat

    @metadata_property(metadata=MetadataConstants.OPTIMISATION_SPACE)
    def optimisation_space(self) -> str:
        """
        Getter property for the optimisation space name.

        Returns
        -------
        :class:`str`
            Optimisation space name
        """

        return self._optimisation_space

    @metadata_property(metadata=MetadataConstants.ILLUMINANT_INTERPOLATOR)
    def illuminant_interpolator(self) -> str:
        """
        Getter property for the illuminant interpolator name.

        Returns
        -------
        :class:`str`
            Illuminant interpolator name.
        """

        return self._illuminant_interpolator

    @metadata_property(metadata=MetadataConstants.DECODING_METHOD)
    def decoding_method(self) -> str:
        """
        Getter property for the decoding method name.

        Returns
        -------
        :class:`str`
            Decoding method name.
        """

        return self._decoding_method

    @metadata_property(metadata=MetadataConstants.EV_RANGE)
    def ev_range(self) -> NDArrayFloat:
        """
        Getter property for the EV range.

        Returns
        -------
        :class:`np.ndarray`
            EV range.
        """

        return self._ev_range

    @metadata_property(metadata=MetadataConstants.GREY_CARD_REFERENCE)
    def grey_card_reference(self) -> NDArrayFloat:
        """
        Getter property for the grey card reference values.

        Returns
        -------
        :class:`np.ndarray`
            Grey card reference values.
        """

        return self._grey_card_reference

    @metadata_property(metadata=MetadataConstants.LUT_SIZE)
    def lut_size(self) -> int:
        """
        Getter property for the *LUT* size.

        Returns
        -------
        :class:`int`
            Size of the *LUT*.
        """

        return self._lut_size

    @metadata_property(metadata=MetadataConstants.LUT_SMOOTHING)
    def lut_smoothing(self) -> int:
        """
        Getter property for the *LUT* smoothing.

        Returns
        -------
        :class:`int`
            Smoothing amount for the *LUT*.
        """

        return self._lut_smoothing

    @metadata_property(metadata=MetadataConstants.DATA)
    def data(self) -> Dict:
        """
        Getter property for the data structure holding the image sequences.

        Returns
        -------
        :class:`dict`
            The data structure to hold the image sequences
        """

        return self._data

    @metadata_property(metadata=MetadataConstants.WORKING_DIR)
    def working_directory(self) -> str:
        """
        Getter property for the working directory for the project.

        Returns
        -------
        :class:`str`
            Working directory for the project.
        """

        return self._working_directory

    @metadata_property(metadata=MetadataConstants.CLEAN_UP)
    def cleanup(self) -> bool:
        """
        Getter property for whether the working directory should be cleaned up.

        Returns
        -------
        :class:`bool`
            Whether the working directory should be cleaned up.
        """

        return self._cleanup

    @metadata_property(metadata=MetadataConstants.REFERENCE_COLOUR_CHECKER)
    def reference_colour_checker(self) -> str:
        """
        Getter property for the reference colour checker name.

        Returns
        -------
        :class:`str`
            Reference colour checker name.
        """

        return self._reference_colour_checker

    @metadata_property(metadata=MetadataConstants.ILLUMINANT)
    def illuminant(self) -> str:
        """
        Getter property for the illuminant name.

        Returns
        -------
        :class:`str`
            Illuminant name.
        """

        return self._illuminant

    @metadata_property(metadata=MetadataConstants.FILE_TYPE)
    def file_type(self) -> str:
        """
        Getter property for the file type, i.e., file extension.

        Returns
        -------
        :class:`str`
            File type, i.e., file extension.
        """

        return self._file_type

    @metadata_property(metadata=MetadataConstants.EV_WEIGHTS)
    def ev_weights(self) -> NDArrayFloat:
        """
        Getter property for the *EV* weights.

        Returns
        -------
        :class:`np.ndarray`
            *EV* weights.
        """

        return self._ev_weights

    @metadata_property(metadata=MetadataConstants.OPTIMIZATION_KWARGS)
    def optimization_kwargs(self) -> Dict:
        """
        Getter property for the optimization keyword arguments.

        Returns
        -------
        :class:`dict`
            Optimization keyword arguments.
        """

        return self._optimization_kwargs

    @metadata_property(metadata=MetadataConstants.INCLUDE_WHITE_BALANCE_IN_CLF)
    def include_white_balance_in_clf(self) -> bool:
        """
        Getter property for whether to include the white balance in the *CLF*.

        Returns
        -------
        :class:`bool`
            Whether to include the white balance in the *CLF*.
        """
        return self._include_white_balance_in_clf

    @metadata_property(metadata=MetadataConstants.FLATTEN_CLF)
    def flatten_clf(self) -> str:
        """
        Getter property for the flatten_clf.

        Returns
        -------
        :class:`bool`
            Whether to flatten the clf output into a 1D LUT and a single 3x3 Matrix
        """

        return self._flatten_clf

    @metadata_property(metadata=MetadataConstants.INCLUDE_EXPOSURE_FACTOR_IN_CLF)
    def include_exposure_factor_in_clf(self) -> bool:
        """
        Getter property for whether to include the exposure factor (K) in the *CLF*.

        Returns
        -------
        :class:`bool`
            Whether to include the exposure factor (K) in the *CLF*.
        """

        return self._include_exposure_factor_in_clf

    def get_reference_colour_checker_samples(self) -> NDArrayFloat:
        """
        Return the reference colour checker samples.

        Returns
        -------
        :class:`np.ndarray`
            Reference colour checker samples.
        """

        # Pass illuminant_custom_temperature if illuminant is "Blackbody" or "Daylight"
        temperature = (
            self.illuminant_custom_temperature
            if self.illuminant in ("Blackbody", "Daylight")
            else None
        )

        return generate_reference_colour_checker(
            get_sds_colour_checker(self.reference_colour_checker),
            get_sds_illuminant(self.illuminant, temperature=temperature),
        )

    def get_optimization_factory(self) -> callable:
        """
        Return the optimisation factory based on the optimisation space.

        Returns
        -------
        :class:`callable`
            Optimisation factory.
        """

        factory = OPTIMISATION_FACTORIES.get(self.optimisation_space)

        if factory is None:
            exception = f'Optimisation space "{self.optimisation_space}" was not found!'

            raise ValueError(exception)

        return factory

    def update(self, value: IDTProjectSettings) -> None:
        """
        Update the project settings with the given value from another project
        settings.

        Parameters
        ----------
        value
            The project settings to update with.
        """

        if not isinstance(value, IDTProjectSettings):
            msg = f'Expected an "IDTProjectSettings" type, got "{type(value)}" instead!'
            raise TypeError(msg)

        for name, prop in value.properties:
            setattr(self, name, prop.getter(value))

    @classmethod
    def from_directory(cls, directory: str) -> IDTProjectSettings:
        """
        Create a new project settings for a given directory on disk and build
        the data structure based on the files on disk.

        Parameters
        ----------
        directory
            The directory to the project root containing the image sequence
            directories.
        """

        instance = cls()
        data = {
            DirectoryStructure.COLOUR_CHECKER: {},
            DirectoryStructure.GREY_CARD: [],
        }

        # Validate folder paths for colour_checker and grey_card exist
        colour_checker_path = os.path.join(
            directory, DirectoryStructure.DATA, DirectoryStructure.COLOUR_CHECKER
        )
        grey_card_path = os.path.join(
            directory, DirectoryStructure.DATA, DirectoryStructure.GREY_CARD
        )

        if not os.path.exists(colour_checker_path) or not os.path.exists(
            grey_card_path
        ):
            msg = 'Required "colour_checker" or "grey_card" folder does not exist.'
            raise ValueError(msg)

        # Populate colour_checker data
        for root, _, files in os.walk(colour_checker_path):
            for file in files:
                exposure_value = os.path.basename(
                    root
                )  # Assuming folder names are the exposure values
                if os.path.basename(file).startswith("."):
                    continue

                if exposure_value not in data[DirectoryStructure.COLOUR_CHECKER]:
                    data[DirectoryStructure.COLOUR_CHECKER][exposure_value] = []
                absolute_file_path = os.path.join(root, file)
                data[DirectoryStructure.COLOUR_CHECKER][exposure_value].append(
                    os.path.relpath(absolute_file_path, start=directory)
                )

        sorted_keys = sorted(
            data[DirectoryStructure.COLOUR_CHECKER], key=sort_exposure_keys
        )

        sorted_colour_checker = {
            format_exposure_key(key): data[DirectoryStructure.COLOUR_CHECKER][key]
            for key in sorted_keys
        }

        data[DirectoryStructure.COLOUR_CHECKER] = sorted_colour_checker

        # Populate grey_card data
        for root, _, files in os.walk(grey_card_path):
            files.sort()
            for file in files:
                if os.path.basename(file).startswith("."):
                    continue
                absolute_file_path = os.path.join(root, file)
                data[DirectoryStructure.GREY_CARD].append(
                    os.path.relpath(absolute_file_path, start=directory)
                )

        instance.data = data
        return instance

    def __str__(self) -> str:
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

        for name, _descriptor in sorted(self.properties):
            attributes.append({"name": f"{name}", "label": f"{name}".title()})

        attributes.append({"line_break": True})

        return multiline_str(self, attributes)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the project settings to ensure that all the minimally required
        inputs are set

        Returns
        -------
        :class:`tuple`
            Whether the minimum required settings are valid or not from the ui and what
            the errors are

        """
        valid, errors = self.validate_core_requirements(
            self.aces_user_name,
            self.encoding_colourspace,
            self.encoding_transfer_function,
            self.camera_make,
            self.camera_model,
            self.aces_transform_id,
        )
        return valid, errors

    @staticmethod
    def validate_core_requirements(
        aces_user_name: str,
        encoding_colourspace: str,
        encoding_transfer_function: str,
        camera_make: str,
        camera_model: str,
        aces_transform_id: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate the key values needed to have a valid project

        Parameters
        ----------
        aces_user_name: str
            the ACES username
        encoding_colourspace: str
            the encoding colourspace
        encoding_transfer_function: str
            the encoding transfer function
        camera_make: str
            the camera make
        camera_model : str
            the camera model
        aces_transform_id: str
            the ACES transform id

        Returns
        -------
        :class:`tuple`
            Whether the minimum required settings are valid or not from the ui and
            what the errors are

        """
        valid = True
        errors = []
        if not camera_make:
            valid = False
            errors.append("No Camera Make Specified")

        if not camera_model:
            valid = False
            errors.append("No Camera Model Specified")

        if not aces_user_name:
            valid = False
            errors.append("No ACES User Name Specified")

        if not encoding_colourspace:
            valid = False
            errors.append("No Encoding Colourspace Specified")

        if not encoding_transfer_function:
            valid = False
            errors.append("No Encoding Transfer Function Specified")

        if not is_valid_csc_urn(aces_transform_id):
            valid = False
            errors.append("Invalid ACES Transform ID")

        return valid, errors
