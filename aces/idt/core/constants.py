"""
IDT Constants
=============

Define the constants for the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import colour

from aces.idt.core.structures import Metadata

if TYPE_CHECKING:
    from colour.hints import Tuple

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "EXPOSURE_CLIPPING_THRESHOLD",
    "DirectoryStructure",
    "UITypes",
    "UICategories",
    "LUTSize",
    "RGBDisplayColourspace",
    "CAT",
    "OptimizationSpace",
    "Interpolators",
    "DecodingMethods",
    "ProjectSettingsMetadataConstants",
]

EXPOSURE_CLIPPING_THRESHOLD: int = 2 / (2**10 - 1)
"""
The threshold for determining if RGB values are clipped. The threshold is
2 code values in a 10 bit system
"""


class DirectoryStructure:
    """Constants for the directory names which compose the data structure."""

    DATA: ClassVar[str] = "data"
    COLOUR_CHECKER: ClassVar[str] = "colour_checker"
    GREY_CARD: ClassVar[str] = "grey_card"
    FLATFIELD: ClassVar[str] = "flatfield"
    WHITE: ClassVar[str] = "white"
    BLACK: ClassVar[str] = "black"


class UITypes:
    """Constants for the UI categories."""

    INT_FIELD: ClassVar[str] = "IntField"
    STRING_FIELD: ClassVar[str] = "StringField"
    OPTIONS_FIELD: ClassVar[str] = "OptionsField"
    VECTOR3_FIELD: ClassVar[str] = "Vector3Field"
    FOLDER_STRUCTURE: ClassVar[str] = "FOLDER_STRUCTURE"
    BOOLEAN_FIELD: ClassVar[str] = "BooleanField"
    ARRAY_FIELD: ClassVar[str] = "ArrayField"
    MAP_FIELD: ClassVar[str] = "MapField"


class UICategories:
    """Constants for the UI categories."""

    ADVANCED: ClassVar[str] = "Advanced"
    STANDARD: ClassVar[str] = "Standard"
    HIDDEN: ClassVar[str] = "HIDDEN"


class LUTSize:
    """Constants for the LUT sizes."""

    LUT_1024: ClassVar[int] = 1024
    LUT_2048: ClassVar[int] = 2048
    LUT_4096: ClassVar[int] = 4096
    LUT_8192: ClassVar[int] = 8192
    LUT_16384: ClassVar[int] = 16384
    LUT_32768: ClassVar[int] = 32768
    LUT_65536: ClassVar[int] = 65536
    DEFAULT: ClassVar[int] = LUT_1024
    ALL: ClassVar[tuple[int, ...]] = (
        LUT_1024,
        LUT_2048,
        LUT_4096,
        LUT_8192,
        LUT_16384,
        LUT_32768,
        LUT_65536,
    )


class RGBDisplayColourspace:
    """Constants for the RGB display colourspaces."""

    SRGB: ClassVar[str] = "sRGB"
    DCI_P3: ClassVar[str] = "Display P3"
    DEFAULT: ClassVar[str] = SRGB
    ALL: ClassVar[tuple[str, ...]] = (SRGB, DCI_P3)


class CATMeta(type):
    @property
    def ALL(cls) -> Tuple[str, ...]:
        """
        Return all the available chromatic adaptation transforms.

        Returns
        -------
        :class:`tuple`
            Available chromatic adaptation transforms.
        """

        return *sorted(colour.CHROMATIC_ADAPTATION_TRANSFORMS), "None"


class CAT(metaclass=CATMeta):
    """Constants for the chromatic adaptation transforms."""

    DEFAULT: ClassVar[str] = "CAT02"


class OptimizationSpace:
    """Constants for the optimization spaces."""

    OKLAB: ClassVar[str] = "Oklab"
    JZAZBZ: ClassVar[str] = "JzAzBz"
    IPT: ClassVar[str] = "IPT"
    CIE_LAB: ClassVar[str] = "CIE Lab"
    DEFAULT: ClassVar[str] = OKLAB
    ALL: ClassVar[tuple[str, ...]] = (OKLAB, JZAZBZ, IPT, CIE_LAB)


class Interpolators:
    """Constants for the interpolators."""

    CUBIC_SPLINE: ClassVar[str] = "Cubic Spline"
    LINEAR: ClassVar[str] = "Linear"
    PCHIP: ClassVar[str] = "PCHIP"
    SPRAGUE_1880: ClassVar[str] = "Sprague (1880)"
    DEFAULT: ClassVar[str] = LINEAR
    ALL: ClassVar[tuple[str, ...]] = (CUBIC_SPLINE, LINEAR, PCHIP, SPRAGUE_1880)


class DecodingMethods:
    """Decoding methods."""

    MEDIAN: ClassVar[str] = "Median"
    AVERAGE: ClassVar[str] = "Average"
    PER_CHANNEL: ClassVar[str] = "Per Channel"
    ACES: ClassVar[str] = "ACES"
    DEFAULT: ClassVar[str] = MEDIAN
    ALL: ClassVar[tuple[str, ...]] = (MEDIAN, AVERAGE, PER_CHANNEL, ACES)


class ProjectSettingsMetadataConstants:
    """Constants for the project settings."""

    SCHEMA_VERSION = Metadata(
        name="schema_version",
        default_value="0.1.0",
        description="The project settings schema version",
        display_name="Schema Version",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    CAMERA_MAKE = Metadata(
        name="camera_make",
        default_value="",
        description="The make of the camera used to capture the images",
        display_name="Camera Make",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    CAMERA_MODEL = Metadata(
        name="camera_model",
        default_value="",
        description="The model of the camera used to capture the images",
        display_name="Camera Model",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    RGB_DISPLAY_COLOURSPACE = Metadata(
        name="rgb_display_colourspace",
        default_value=RGBDisplayColourspace.DEFAULT,
        description="The RGB display colourspace",
        display_name="RGB Display Colourspace",
        ui_type=UITypes.OPTIONS_FIELD,
        options=RGBDisplayColourspace.ALL,
        ui_category=UICategories.ADVANCED,
    )

    CAT = Metadata(
        name="cat",
        default_value=CAT.DEFAULT,
        description="The CAT",
        display_name="CAT",
        ui_type=UITypes.OPTIONS_FIELD,
        options=CAT.ALL,
        ui_category=UICategories.ADVANCED,
    )

    OPTIMISATION_SPACE = Metadata(
        name="optimisation_space",
        default_value=OptimizationSpace.DEFAULT,
        description="The optimisation space",
        display_name="Optimisation Space",
        ui_type=UITypes.OPTIONS_FIELD,
        options=OptimizationSpace.ALL,
        ui_category=UICategories.ADVANCED,
    )

    ILLUMINANT_INTERPOLATOR = Metadata(
        name="illuminant_interpolator",
        default_value=Interpolators.DEFAULT,
        description="The illuminant interpolator",
        display_name="Illuminant Interpolator",
        ui_type=UITypes.STRING_FIELD,
        options=Interpolators.ALL,
        ui_category=UICategories.ADVANCED,
    )

    DECODING_METHOD = Metadata(
        name="decoding_method",
        default_value=DecodingMethods.DEFAULT,
        description="The decoding method",
        display_name="Decoding Method",
        ui_type=UITypes.STRING_FIELD,
        options=DecodingMethods.ALL,
        ui_category=UICategories.ADVANCED,
    )

    EV_RANGE = Metadata(
        name="ev_range",
        default_value=[-1.0, 0.0, 1.0],
        description="The EV range",
        display_name="EV Range",
        ui_type=UITypes.ARRAY_FIELD,
        ui_category=UICategories.ADVANCED,
    )

    GREY_CARD_REFERENCE = Metadata(
        name="grey_card_reference",
        default_value=[0.18, 0.18, 0.18],
        description="The grey card reference",
        display_name="Grey Card Reference",
        ui_type=UITypes.VECTOR3_FIELD,
        ui_category=UICategories.ADVANCED,
    )

    LUT_SIZE = Metadata(
        name="lut_size",
        default_value=LUTSize.DEFAULT,
        description="The LUT size",
        display_name="LUT Size",
        ui_type=UITypes.OPTIONS_FIELD,
        options=LUTSize.ALL,
        ui_category=UICategories.STANDARD,
    )

    LUT_SMOOTHING = Metadata(
        name="lut_smoothing",
        default_value=16,
        description="The LUT smoothing",
        display_name="LUT Smoothing",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ACES_TRANSFORM_ID = Metadata(
        name="aces_transform_id",
        default_value="",
        description="The ACES transform ID",
        display_name="ACES Transform ID",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ACES_USER_NAME = Metadata(
        name="aces_user_name",
        default_value="",
        description="The ACES username",
        display_name="ACES Username",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ISO = Metadata(
        name="iso",
        default_value=800,
        description="The ISO",
        display_name="ISO",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    TEMPERATURE = Metadata(
        name="temperature",
        default_value=6000,
        description="The temperature",
        display_name="Temperature",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ADDITIONAL_CAMERA_SETTINGS = Metadata(
        name="additional_camera_settings",
        default_value="",
        description="The additional camera settings",
        display_name="Additional Camera Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    LIGHTING_SETUP_DESCRIPTION = Metadata(
        name="lighting_setup_description",
        default_value="",
        description="The lighting setup description",
        display_name="Lighting Setup Description",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DEBAYERING_PLATFORM = Metadata(
        name="debayering_platform",
        default_value="",
        description="The debayering platform",
        display_name="Debayering Platform",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DEBAYERING_SETTINGS = Metadata(
        name="debayering_settings",
        default_value="",
        description="The debayering settings",
        display_name="Debayering Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ENCODING_COLOUR_SPACE = Metadata(
        name="encoding_colourspace",
        default_value="",
        description="The encoding colourspace",
        display_name="Encoding Colourspace",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ENCODING_TRANSFER_FUNCTION = Metadata(
        name="encoding_transfer_function",
        default_value="",
        description="The encoding transfer function",
        display_name="Encoding Transfer Function",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DATA = Metadata(
        name="data",
        default_value={
            DirectoryStructure.COLOUR_CHECKER: {},
            DirectoryStructure.GREY_CARD: {},
        },
        description="The folder structure for the ",
        display_name="Folder Structure",
        ui_type=UITypes.FOLDER_STRUCTURE,
        serialize_group="",
        ui_category=UICategories.HIDDEN,
    )

    WORKING_DIR = Metadata(
        name="working_directory",
        default_value="",
        description="The file path to the working directory",
        display_name="Working Directory",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    CLEAN_UP = Metadata(
        name="cleanup",
        default_value=True,
        description="Do we want to cleanup the working directory after we finish?",
        display_name="Cleanup",
        ui_type=UITypes.BOOLEAN_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    REFERENCE_COLOUR_CHECKER = Metadata(
        name="reference_colour_checker",
        default_value="ISO 17321-1",
        description="The reference colour checker we want to use",
        ui_type=UITypes.OPTIONS_FIELD,
        options=sorted(colour.SDS_COLOURCHECKERS),
        ui_category=UICategories.ADVANCED,
    )

    ILLUMINANT = Metadata(
        name="illuminant",
        default_value="D60",
        description="The illuminant we want to use for the reference colour checker",
        ui_type=UITypes.OPTIONS_FIELD,
        options=["Custom", "Daylight", "Blackbody", *sorted(colour.SDS_ILLUMINANTS)],
        ui_category=UICategories.ADVANCED,
    )

    FILE_TYPE = Metadata(
        name="file_type",
        default_value="",
        description="The file type of the recorded footage is detected from "
        "the archive",
        display_name="File Type",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    EV_WEIGHTS = Metadata(
        name="ev_weights",
        default_value=[],
        description="Normalised weights used to sum the exposure values. If not given,"
        "the median of the exposure values is used.",
        display_name="EV Weights",
        ui_type=UITypes.ARRAY_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    OPTIMIZATION_KWARGS = Metadata(
        name="optimization_kwargs",
        default_value={},
        description="Parameters for the optimization function scipy.optimize.minimize",
        display_name="Optimization Kwargs",
        ui_type=UITypes.MAP_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    INCLUDE_WHITE_BALANCE_IN_CLF = Metadata(
        name="include_white_balance_in_clf",
        default_value=False,
        description="Whether to include the White Balance Matrix in the CLF",
        display_name="Include White Balance in CLF",
        ui_type=UITypes.BOOLEAN_FIELD,
        ui_category=UICategories.STANDARD,
    )

    FLATTEN_CLF = Metadata(
        name="flatten_clf",
        default_value=False,
        description="Whether to flatten the CLF to a 1D Lut and a single 3x3 Matrix ",
        display_name="Flatten CLF",
        ui_type=UITypes.BOOLEAN_FIELD,
        ui_category=UICategories.STANDARD,
    )

    INCLUDE_EXPOSURE_FACTOR_IN_CLF = Metadata(
        name="include_exposure_factor_in_clf",
        default_value=False,
        description="Whether to include the exposure factor (K) in the CLF",
        display_name="Include Exposure Factor in CLF",
        ui_type=UITypes.BOOLEAN_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ALL: ClassVar[tuple[Metadata, ...]] = (
        SCHEMA_VERSION,
        CAMERA_MAKE,
        CAMERA_MODEL,
        RGB_DISPLAY_COLOURSPACE,
        CAT,
        OPTIMISATION_SPACE,
        ILLUMINANT_INTERPOLATOR,
        DECODING_METHOD,
        EV_RANGE,
        GREY_CARD_REFERENCE,
        LUT_SIZE,
        LUT_SMOOTHING,
        ACES_TRANSFORM_ID,
        ACES_USER_NAME,
        ISO,
        TEMPERATURE,
        ADDITIONAL_CAMERA_SETTINGS,
        LIGHTING_SETUP_DESCRIPTION,
        DEBAYERING_PLATFORM,
        DEBAYERING_SETTINGS,
        ENCODING_COLOUR_SPACE,
        ENCODING_TRANSFER_FUNCTION,
        DATA,
        WORKING_DIR,
        CLEAN_UP,
        REFERENCE_COLOUR_CHECKER,
        ILLUMINANT,
        FILE_TYPE,
        EV_WEIGHTS,
        OPTIMIZATION_KWARGS,
        INCLUDE_WHITE_BALANCE_IN_CLF,
        FLATTEN_CLF,
        INCLUDE_EXPOSURE_FACTOR_IN_CLF,
    )
