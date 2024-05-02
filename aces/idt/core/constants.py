""" MODULE which holds constants for the project

"""
import colour

from .structures import IDTMetaData


class DataFolderStructure:
    DATA = "data"
    COLOUR_CHECKER = "colour_checker"
    GREY_CARD = "grey_card"



class UITypes:
    """ Constants for the UI categories

    """
    INT_FIELD = "IntField"
    STRING_FIELD = "StringField"
    OPTIONS_FIELD = "OptionsField"
    VECTOR3_FIELD = "Vector3Field"
    FOLDER_STRUCTURE = "FOLDER_STRUCTURE"


class UICategories:
    """ Constants for the UI categories

    """
    ADVANCED = "Advanced"
    STANDARD = "Standard"
    HIDDEN = "HIDDEN"


class LUTSize:
    """ Constants for the LUT sizes

    """
    LUT_1024 = 1024
    LUT_2048 = 2048
    LUT_4096 = 4096
    LUT_8192 = 8192
    LUT_16384 = 16384
    LUT_32768 = 32768
    LUT_65536 = 65536
    DEFAULT = LUT_1024
    ALL = [LUT_1024, LUT_2048, LUT_4096, LUT_8192, LUT_16384, LUT_32768, LUT_65536]


class RGBDisplayColourspace:
    """ Constants for the RGB display colourspaces

    """
    SRGB = "sRGB"
    DCI_P3 = "Display P3"
    DEFAULT = SRGB
    ALL = [SRGB, DCI_P3]


class CAT:
    """ Constants for the CATs """
    DEFAULT = "CAT02"

    @classmethod
    @property
    def ALL(cls):
        cats = list(colour.CHROMATIC_ADAPTATION_TRANSFORMS.keys())
        cats.sort()
        return cats


class OptimizationSpace:
    """ Constants for the optimization spaces """
    OKLAB = "Oklab"
    JZAZBZ = "JzAzBz"
    IPT = "IPT"
    CIE_LAB = "CIE Lab"
    DEFAULT = OKLAB
    ALL = [OKLAB, JZAZBZ, IPT, CIE_LAB]


class Interpolators:
    """ Constants for the interpolators
    """
    CUBIC_SPLINE = "Cubic Spline"
    LINEAR = "Linear"
    PCHIP = "PCHIP"
    SPRAGUE_1880 = "Sprague (1880)"
    DEFAULT = LINEAR
    ALL = [CUBIC_SPLINE, LINEAR, PCHIP, SPRAGUE_1880]


class DecodingMethods:
    """ Decoding methods """
    MEDIAN = "Median"
    AVERAGE = "Average"
    PER_CHANNEL = "Per Channel"
    ACES = "ACES"
    DEFAULT = MEDIAN
    ALL = [MEDIAN, AVERAGE, PER_CHANNEL, ACES]


class ProjectSettingsMetaDataConstants:
    """ Constants for the project settings

    """
    SCHEMA_VERSION = IDTMetaData(
        default_value="0.1.0",
        description="The project settings schema version",
        display_name="Schema Version",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.HIDDEN)

    CAMERA_MAKE = IDTMetaData(
        default_value="",
        description="The make of the camera used to capture the images",
        display_name="Camera Make",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD)

    CAMERA_MODEL = IDTMetaData(
        default_value="",
        description="The model of the camera used to capture the images",
        display_name="Camera Model",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    RGB_DISPLAY_COLOURSPACE = IDTMetaData(
        default_value=RGBDisplayColourspace.DEFAULT,
        description="The RGB display colourspace",
        display_name="RGB Display Colourspace",
        ui_type=UITypes.OPTIONS_FIELD,
        options=RGBDisplayColourspace.ALL,
        ui_category=UICategories.ADVANCED
    )

    CAT = IDTMetaData(
        default_value=CAT.DEFAULT,
        description="The CAT",
        display_name="CAT",
        ui_type=UITypes.OPTIONS_FIELD,
        options=CAT.ALL,
        ui_category=UICategories.ADVANCED
    )

    OPTIMISATION_SPACE = IDTMetaData(
        default_value=OptimizationSpace.DEFAULT,
        description="The optimisation space",
        display_name="Optimisation Space",
        ui_type=UITypes.OPTIONS_FIELD,
        options=OptimizationSpace.ALL,
        ui_category=UICategories.ADVANCED
    )

    ILLUMINANT_INTERPOLATOR = IDTMetaData(
        default_value=Interpolators.DEFAULT,
        description="The illuminant interpolator",
        display_name="Illuminant Interpolator",
        ui_type=UITypes.STRING_FIELD,
        options=Interpolators.ALL,
        ui_category=UICategories.ADVANCED
    )

    DECODING_METHOD = IDTMetaData(
        default_value=DecodingMethods.DEFAULT,
        description="The decoding method",
        display_name="Decoding Method",
        ui_type=UITypes.STRING_FIELD,
        options=DecodingMethods.ALL,
        ui_category=UICategories.ADVANCED
    )

    EV_RANGE = IDTMetaData(
        default_value=[-1., 0., 1.],
        description="The EV range",
        display_name="EV Range",
        ui_type=UITypes.VECTOR3_FIELD,
        ui_category=UICategories.ADVANCED
    )

    GREY_CARD_REFERENCE = IDTMetaData(
        default_value=[0.18, 0.18, 0.18],
        description="The grey card reference",
        display_name="Grey Card Reference",
        ui_type=UITypes.VECTOR3_FIELD,
        ui_category=UICategories.ADVANCED
    )

    LUT_SIZE = IDTMetaData(
        default_value=LUTSize.DEFAULT,
        description="The LUT size",
        display_name="LUT Size",
        ui_type=UITypes.OPTIONS_FIELD,
        options=LUTSize.ALL,
        ui_category=UICategories.STANDARD
    )

    LUT_SMOOTHING = IDTMetaData(
        default_value=32,
        description="The LUT smoothing",
        display_name="LUT Smoothing",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD
    )

    ACES_TRANSFORM_ID = IDTMetaData(
        default_value="",
        description="The ACES transform ID",
        display_name="ACES Transform ID",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    ACES_USER_NAME = IDTMetaData(
        default_value="",
        description="The ACES username",
        display_name="ACES Username",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    ISO = IDTMetaData(
        default_value=800,
        description="The ISO",
        display_name="ISO",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    TEMPERATURE = IDTMetaData(
        default_value=5600,
        description="The temperature",
        display_name="Temperature",
        ui_type=UITypes.INT_FIELD,
        ui_category=UICategories.STANDARD
    )

    ADDITIONAL_CAMERA_SETTINGS = IDTMetaData(
        default_value="",
        description="The additional camera settings",
        display_name="Additional Camera Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    LIGHTING_SETUP_DESCRIPTION = IDTMetaData(
        default_value="",
        description="The lighting setup description",
        display_name="Lighting Setup Description",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    DEBAYERING_PLATFORM = IDTMetaData(
        default_value="",
        description="The debayering platform",
        display_name="Debayering Platform",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    DEBAYERING_SETTINGS = IDTMetaData(
        default_value="",
        description="The debayering settings",
        display_name="Debayering Settings",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    ENCODING_COLOUR_SPACE = IDTMetaData(
        default_value="",
        description="The encoding colour space",
        display_name="Encoding Colour Space",
        ui_type=UITypes.STRING_FIELD,
        ui_category=UICategories.STANDARD
    )

    DATA = IDTMetaData(
        default_value={DataFolderStructure.COLOUR_CHECKER: {}, DataFolderStructure.GREY_CARD: {}},
        description="The folder structure for the ",
        display_name="Schema Version",
        ui_type=UITypes.FOLDER_STRUCTURE,
        serialize_group="",
        ui_category=UICategories.HIDDEN)

    ALL = [
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
        DATA
    ]
