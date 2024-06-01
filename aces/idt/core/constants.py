"""Module which holds constants for the project

"""
from typing import ClassVar

import colour
import numpy as np

from .structures import IDTMetaData


class DataFolderStructure:
    """Constants for the folder names which make up the data structure"""

    DATA: ClassVar[str] = "data"
    COLOUR_CHECKER: ClassVar[str] = "colour_checker"
    GREY_CARD: ClassVar[str] = "grey_card"


class UITypes:
    """Constants for the UI categories"""

    INT_FIELD: ClassVar[str] = "IntField"
    STRING_FIELD: ClassVar[str] = "StringField"
    OPTIONS_FIELD: ClassVar[str] = "OptionsField"
    VECTOR3_FIELD: ClassVar[str] = "Vector3Field"
    FOLDER_STRUCTURE: ClassVar[str] = "FOLDER_STRUCTURE"
    BOOLEAN_FIELD: ClassVar[str] = "BooleanField"
    ARRAY_FIELD: ClassVar[str] = "ArrayField"
    MAP_FIELD: ClassVar[str] = "MapField"


class UICategories:
    """Constants for the UI categories"""

    ADVANCED: ClassVar[str] = "Advanced"
    STANDARD: ClassVar[str] = "Standard"
    HIDDEN: ClassVar[str] = "HIDDEN"


class LUTSize:
    """Constants for the LUT sizes"""

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
    """Constants for the RGB display colourspaces"""

    SRGB: ClassVar[str] = "sRGB"
    DCI_P3: ClassVar[str] = "Display P3"
    DEFAULT: ClassVar[str] = SRGB
    ALL: ClassVar[tuple[str, ...]] = (SRGB, DCI_P3)


class CAT:
    """Constants for the CATs"""

    DEFAULT: ClassVar[str] = "CAT02"

    @classmethod
    @property
    def ALL(cls):
        """

        Returns A list of all the CATs
        -------

        """
        cats = list(colour.CHROMATIC_ADAPTATION_TRANSFORMS.keys())
        cats.sort()
        return cats


class OptimizationSpace:
    """Constants for the optimization spaces"""

    OKLAB: ClassVar[str] = "Oklab"
    JZAZBZ: ClassVar[str] = "JzAzBz"
    IPT: ClassVar[str] = "IPT"
    CIE_LAB: ClassVar[str] = "CIE Lab"
    DEFAULT: ClassVar[str] = OKLAB
    ALL: ClassVar[tuple[str, ...]] = (OKLAB, JZAZBZ, IPT, CIE_LAB)


class Interpolators:
    """Constants for the interpolators"""

    CUBIC_SPLINE: ClassVar[str] = "Cubic Spline"
    LINEAR: ClassVar[str] = "Linear"
    PCHIP: ClassVar[str] = "PCHIP"
    SPRAGUE_1880: ClassVar[str] = "Sprague (1880)"
    DEFAULT: ClassVar[str] = LINEAR
    ALL: ClassVar[tuple[str, ...]] = (CUBIC_SPLINE, LINEAR, PCHIP, SPRAGUE_1880)


class DecodingMethods:
    """Decoding methods"""

    MEDIAN: ClassVar[str] = "Median"
    AVERAGE: ClassVar[str] = "Average"
    PER_CHANNEL: ClassVar[str] = "Per Channel"
    ACES: ClassVar[str] = "ACES"
    DEFAULT: ClassVar[str] = MEDIAN
    ALL: ClassVar[tuple[str, ...]] = (MEDIAN, AVERAGE, PER_CHANNEL, ACES)


class ProjectSettingsMetaDataConstants:
    """Constants for the project settings"""

    SCHEMA_VERSION = IDTMetaData(
        default_value="0.1.0",
        description="The project settings schema version",
        display_name="Schema Version",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    CAMERA_MAKE = IDTMetaData(
        default_value="",
        description="The make of the camera used to capture the images",
        display_name="Camera Make",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    CAMERA_MODEL = IDTMetaData(
        default_value="",
        description="The model of the camera used to capture the images",
        display_name="Camera Model",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    RGB_DISPLAY_COLOURSPACE = IDTMetaData(
        default_value=RGBDisplayColourspace.DEFAULT,
        description="The RGB display colourspace",
        display_name="RGB Display Colourspace",
        ui_type=UITypes.OPTIONS_FIELD,
        options=RGBDisplayColourspace.ALL,
        ui_category=UICategories.ADVANCED,
    )

    CAT = IDTMetaData(
        default_value=CAT.DEFAULT,
        description="The CAT",
        display_name="CAT",
        ui_type=UITypes.OPTIONS_FIELD,
        options=CAT.ALL,
        ui_category=UICategories.ADVANCED,
    )

    OPTIMISATION_SPACE = IDTMetaData(
        default_value=OptimizationSpace.DEFAULT,
        description="The optimisation space",
        display_name="Optimisation Space",
        ui_type=UITypes.OPTIONS_FIELD,
        options=OptimizationSpace.ALL,
        ui_category=UICategories.ADVANCED,
    )

    ILLUMINANT_INTERPOLATOR = IDTMetaData(
        default_value=Interpolators.DEFAULT,
        description="The illuminant interpolator",
        display_name="Illuminant Interpolator",
        ui_type=UITypes.STRING_FIELD,
        options=Interpolators.ALL,
        ui_category=UICategories.ADVANCED,
    )

    DECODING_METHOD = IDTMetaData(
        default_value=DecodingMethods.DEFAULT,
        description="The decoding method",
        display_name="Decoding Method",
        ui_type=UITypes.STRING_FIELD,
        options=DecodingMethods.ALL,
        ui_category=UICategories.ADVANCED,
    )

    EV_RANGE = IDTMetaData(
        default_value=[-1.0, 0.0, 1.0],
        description="The EV range",
        display_name="EV Range",
        ui_type=UITypes.VECTOR3_FIELD,
        ui_category=UICategories.ADVANCED,
    )

    GREY_CARD_REFERENCE = IDTMetaData(
        default_value=[0.18, 0.18, 0.18],
        description="The grey card reference",
        display_name="Grey Card Reference",
        ui_type=UITypes.VECTOR3_FIELD,
        ui_category=UICategories.ADVANCED,
    )

    LUT_SIZE = IDTMetaData(
        default_value=LUTSize.DEFAULT,
        description="The LUT size",
        display_name="LUT Size",
        ui_type=UITypes.OPTIONS_FIELD,
        options=LUTSize.ALL,
        ui_category=UICategories.STANDARD,
    )

    LUT_SMOOTHING = IDTMetaData(
        default_value=32,
        description="The LUT smoothing",
        display_name="LUT Smoothing",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ACES_TRANSFORM_ID = IDTMetaData(
        default_value="",
        description="The ACES transform ID",
        display_name="ACES Transform ID",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ACES_USER_NAME = IDTMetaData(
        default_value="",
        description="The ACES username",
        display_name="ACES Username",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ISO = IDTMetaData(
        default_value=800,
        description="The ISO",
        display_name="ISO",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    TEMPERATURE = IDTMetaData(
        default_value=6000,
        description="The temperature",
        display_name="Temperature",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ADDITIONAL_CAMERA_SETTINGS = IDTMetaData(
        default_value="",
        description="The additional camera settings",
        display_name="Additional Camera Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    LIGHTING_SETUP_DESCRIPTION = IDTMetaData(
        default_value="",
        description="The lighting setup description",
        display_name="Lighting Setup Description",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DEBAYERING_PLATFORM = IDTMetaData(
        default_value="",
        description="The debayering platform",
        display_name="Debayering Platform",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DEBAYERING_SETTINGS = IDTMetaData(
        default_value="",
        description="The debayering settings",
        display_name="Debayering Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    ENCODING_COLOUR_SPACE = IDTMetaData(
        default_value="",
        description="The encoding colour space",
        display_name="Encoding Colour Space",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    DATA = IDTMetaData(
        default_value={
            DataFolderStructure.COLOUR_CHECKER: {},
            DataFolderStructure.GREY_CARD: {},
        },
        description="The folder structure for the ",
        display_name="Folder Structure",
        ui_type=UITypes.FOLDER_STRUCTURE,
        serialize_group="",
        ui_category=UICategories.HIDDEN,
    )

    WORKING_DIR = IDTMetaData(
        default_value="",
        description="The file path to the working directory",
        display_name="Working Directory",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    CLEAN_UP = IDTMetaData(
        default_value=False,
        description="Do we want to cleanup the directory after we finish",
        display_name="Cleanup",
        ui_type=UITypes.BOOLEAN_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    # TODO Thomas, there must be a list of the options available in colour somewhere
    REFERENCE_COLOUR_CHECKER = IDTMetaData(
        default_value="ISO 17321-1",
        description="The reference colour checker we want to use",
        ui_type=UITypes.OPTIONS_FIELD,
        options=["ISO 17321-1"],
        ui_category=UICategories.ADVANCED,
    )

    # TODO Thomas, there must be a list of the illuminants available in colour somewhere
    ILLUMINANT = IDTMetaData(
        default_value="D60",
        description="The illuminant we want to use for the reference colour checker",
        ui_type=UITypes.OPTIONS_FIELD,
        options=["D60"],
        ui_category=UICategories.ADVANCED,
    )

    SIGMA = IDTMetaData(
        default_value=16,
        description="The sigma value used",
        display_name="Sigma",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD,
    )

    FILE_TYPE = IDTMetaData(
        default_value="",
        description="The file type of the recorded footage is detected from "
        "the archive",
        display_name="File Type",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD,
    )

    EV_WEIGHTS = IDTMetaData(
        default_value=np.array([]),
        description="Normalised weights used to sum the exposure values. If not given,"
        "the median of the exposure values is used.",
        display_name="EV Weights",
        ui_type=UITypes.ARRAY_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    OPTIMIZATION_KWARGS = IDTMetaData(
        default_value={},
        description="Parameters for the optimization function scipy.optimize.minimize",
        display_name="Optimization Kwargs",
        ui_type=UITypes.MAP_FIELD,
        ui_category=UICategories.HIDDEN,
    )

    ALL: ClassVar[tuple[IDTMetaData, ...]] = (
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
        DATA,
        WORKING_DIR,
        CLEAN_UP,
        REFERENCE_COLOUR_CHECKER,
        ILLUMINANT,
        SIGMA,
        FILE_TYPE,
        EV_WEIGHTS,
        OPTIMIZATION_KWARGS,
    )
