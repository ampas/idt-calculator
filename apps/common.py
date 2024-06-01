"""
Common Apps Utilities
=====================
"""

import xml.etree.ElementTree as Et

import colour
import colour_checker_detection  # noqa: F401
import colour_datasets
import numpy as np
from colour import (
    CubicSplineInterpolator,
    LinearInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
)
from dash_bootstrap_components import (
    Card,
    CardBody,
    CardHeader,
)
from dash_bootstrap_components import Input as Field
from dash_bootstrap_components import (
    InputGroup,
    InputGroupText,
    Tooltip,
)

from aces.idt import (
    clf_processing_elements,
)

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "COLOUR_ENVIRONMENT",
    "metadata_card_default",
    "DATATABLE_DECIMALS",
    "CUSTOM_WAVELENGTHS",
    "DATASET_RAW_TO_ACES",
    "TRAINING_DATA_KODAK190PATCHES",
    "MSDS_CAMERA_SENSITIVITIES",
    "OPTIONS_CAMERA_SENSITIVITIES",
    "OPTIONS_CAT",
    "OPTIONS_ILLUMINANT",
    "OPTIONS_INTERPOLATION",
    "INTERPOLATORS",
    "OPTIONS_OPTIMISATION_SPACES",
    "OPTIONS_DISPLAY_COLOURSPACES",
    "DELAY_TOOLTIP_DEFAULT",
    "TEMPLATE_DEFAULT_OUTPUT",
    "TEMPLATE_NUKE_GROUP",
    "TEMPLATE_CTL_MODULE",
    "TEMPLATE_DCTL_MODULE",
    "format_float",
    "format_matrix_nuke",
    "format_vector_nuke",
    "format_matrix_ctl",
    "format_vector_ctl",
    "format_float_dctl",
    "format_matrix_dctl",
    "format_vector_dctl",
    "format_idt_clf",
]

from aces.idt.core.common import OPTIMISATION_FACTORIES

COLOUR_ENVIRONMENT = None
"""
*Colour* environment formatted as a string.

COLOUR_ENVIRONMENT : str
"""


def _print_colour_environment(describe):
    """
    Intercept colour environment description output to save it as a formatted
    strings.
    """

    global COLOUR_ENVIRONMENT  # noqa: PLW0603

    if COLOUR_ENVIRONMENT is None:
        COLOUR_ENVIRONMENT = ""

    print(describe)  # noqa: T201

    COLOUR_ENVIRONMENT += describe
    COLOUR_ENVIRONMENT += "\n"


colour.utilities.describe_environment(print_callable=_print_colour_environment)


def metadata_card_default(_uid, *args):
    """
    Return the default metadata card for an application.

    Parameters
    ----------
    _uid : Callable
        Callable to generate a unique id for given id by appending the
        application *UID*.

    Other Parameters
    ----------------
    \\*args
        Optional children.

    Returns
    -------
    :class:`dash_bootstrap_components.Card`
        Metadata card.
    """

    return Card(
        [
            CardHeader("Metadata"),
            CardBody(
                [
                    InputGroup(
                        [
                            InputGroupText("ACEStransformID"),
                            Field(
                                id=_uid("acestransformid-field"),
                                type="text",
                                placeholder="...",
                                debounce=True,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        '"ACEStransformID" of the IDT, e.g. '
                        '"urn:ampas:aces:transformId:v1.5:IDT.ARRI.ARRI-LogC4.a1.v1"',
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("acestransformid-field"),
                    ),
                    InputGroup(
                        [
                            InputGroupText("ACESuserName"),
                            Field(
                                id=_uid("acesusername-field"),
                                type="text",
                                placeholder="...",
                                debounce=True,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        '"ACESuserName" of the IDT, e.g. '
                        '"ACES 1.0 Input - ARRI LogC4"',
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("acesusername-field"),
                    ),
                    InputGroup(
                        [
                            InputGroupText("Camera Make"),
                            Field(
                                id=_uid("camera-make-field"),
                                type="text",
                                placeholder="...",
                                debounce=True,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        'Manufacturer of the camera, e.g. "ARRI" or "RED"',
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("camera-make-field"),
                    ),
                    InputGroup(
                        [
                            InputGroupText("Camera Model"),
                            Field(
                                id=_uid("camera-model-field"),
                                type="text",
                                placeholder="...",
                                debounce=True,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        (
                            'Model of the camera, e.g. "ALEXA 35" or '
                            '"V-RAPTOR XL 8K VV"'
                        ),
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("camera-model-field"),
                    ),
                    *list(args),
                ]
            ),
        ],
        className="mb-2",
    )


DATATABLE_DECIMALS = 7
"""
Datatable decimals.

DATATABLE_DECIMALS : int
"""

CUSTOM_WAVELENGTHS = [*list(range(380, 395, 5)), "..."]
"""
Custom wavelengths list.

CUSTOM_WAVELENGTHS : list
"""

DATASET_RAW_TO_ACES = colour_datasets.load(
    "RAW to ACES Utility Data - Dyer et al. (2017)"
)
"""
*RAW to ACES* dataset.

DATASET_RAW_TO_ACES : dict
"""

TRAINING_DATA_KODAK190PATCHES = DATASET_RAW_TO_ACES["training"]["190-patch"]
"""
*Kodak* 190 patches training data.

TRAINING_DATA_KODAK190PATCHES : MultiSpectralDistributions
"""

MSDS_CAMERA_SENSITIVITIES = DATASET_RAW_TO_ACES["camera"]
"""
Camera sensitivities multi-spectral distributions.

MSDS_CAMERA_SENSITIVITIES : dict
"""

OPTIONS_CAMERA_SENSITIVITIES = [
    {"label": key, "value": key} for key in sorted(MSDS_CAMERA_SENSITIVITIES)
]
"""
Camera sensitivities options for a :class:`Dropdown` class instance.

OPTIONS_CAMERA_SENSITIVITIES : list
"""
OPTIONS_CAMERA_SENSITIVITIES.insert(0, {"label": "Custom", "value": "Custom"})

OPTIONS_CAT = [
    {"label": key, "value": key}
    for key in sorted(colour.CHROMATIC_ADAPTATION_TRANSFORMS.keys())
]
"""
*Chromatic adaptation transform* options for a :class:`Dropdown` class
instance.

OPTIONS_CAT : list
"""
OPTIONS_CAT.append({"label": "None", "value": None})

OPTIONS_ILLUMINANT = [
    {"label": key, "value": key} for key in sorted(colour.SDS_ILLUMINANTS.keys())
]
"""
Illuminant options for a :class:`Dropdown`class instance.

ILLUMINANTS_OPTIONS : list
"""
OPTIONS_ILLUMINANT.insert(0, {"label": "Custom", "value": "Custom"})
OPTIONS_ILLUMINANT.insert(1, {"label": "Blackbody", "value": "Blackbody"})
OPTIONS_ILLUMINANT.insert(1, {"label": "Daylight", "value": "Daylight"})

OPTIONS_INTERPOLATION = [
    {"label": key, "value": key}
    for key in ["Cubic Spline", "Linear", "PCHIP", "Sprague (1880)"]
]

INTERPOLATORS = {
    "Cubic Spline": CubicSplineInterpolator,
    "Linear": LinearInterpolator,
    "PCHIP": PchipInterpolator,
    "Sprague (1880)": SpragueInterpolator,
}
"""
Spectral distribution interpolators.

INTERPOLATORS : dict
"""

OPTIONS_OPTIMISATION_SPACES = [
    {"label": key, "value": key} for key in OPTIMISATION_FACTORIES
]
"""
Optimisation colourspaces.

OPTIONS_OPTIMISATION_SPACES : list
"""

OPTIONS_DISPLAY_COLOURSPACES = [
    {"label": key, "value": key} for key in ["sRGB", "Display P3"]
]
"""
Display colourspaces.

OPTIONS_DISPLAY_COLOURSPACES : list
"""


DELAY_TOOLTIP_DEFAULT = {"show": 500, "hide": 125}
"""
Default tooltip delay settings.

DELAY_TOOLTIP_DEFAULT : list
"""

TEMPLATE_DEFAULT_OUTPUT = """
IDT Matrix
----------

{0}

White Balance Multipliers
-------------------------

{1}"""[1:]
"""
Default formatting template.

TEMPLATE_DEFAULT_OUTPUT : str
"""

TEMPLATE_NUKE_GROUP = """
Group {{
 name {group}_Input_Device_Transform
 tile_color 0xffbf00ff
 xpos 0
 ypos 0
 addUserKnob {{20 idt_Tab l "Input Device Transform"}}
 addUserKnob {{7 k_Floating_Point_Slider l "Exposure Factor"}}
 k_Floating_Point_Slider {k_factor}
 addUserKnob {{26 ""}}
 addUserKnob {{41 idt_matrix l "IDT Matrix" T B_ColorMatrix.matrix}}
 addUserKnob {{41 b_RGB_Color_Knob l "White Balance Multipliers" \
T Exposure_White_Balance_Expression.b_RGB_Color_Knob}}
 addUserKnob {{20 about_Tab l About}}
 addUserKnob {{26 description_Text l "" +STARTLINE T "\
Input Device Transform (IDT)\
\n\nComputed with {application}\
\nUrl : {url}\
\nCamera Make : {camera_make}\
\nCamera Model : {camera_model}\
\nACEStransformID : {aces_transform_id}\
\nACESuserName : {aces_user_name}\
\nScene adopted white : {illuminant}\
\nInput : Linear Camera RGB\
\nOutput : ACES 2065-1\
\nGenerated on : {date}"}}
}}
 Input {{
  inputs 0
  name Input
  xpos 0
 }}
 Expression {{
  temp_name0 k
  temp_expr0 parent.k_Floating_Point_Slider
  temp_name1 min_b
  temp_expr1 "min(b_RGB_Color_Knob.r, b_RGB_Color_Knob.g, b_RGB_Color_Knob.b)"
  expr0 "clamp((b_RGB_Color_Knob.r * r * k) / min_b)"
  expr1 "clamp((b_RGB_Color_Knob.g * g * k) / min_b)"
  expr2 "clamp((b_RGB_Color_Knob.b * b * k) / min_b)"
  name Exposure_White_Balance_Expression
  xpos 0
  ypos 25
  addUserKnob {{20 white_balance_Tab l "White Balance"}}
  addUserKnob {{18 b_RGB_Color_Knob l b}}
  b_RGB_Color_Knob {{ {multipliers} }}
  addUserKnob {{6 b_RGB_Color_Knob_panelDropped l "panel dropped state" -STARTLINE +HIDDEN}}
 }}
 ColorMatrix {{
  matrix {{
      {matrix}
    }}
  name B_ColorMatrix
  xpos 0
  ypos 50
 }}
 Output {{
  name Output
  xpos 0
  ypos 75
 }}
end_group
""".strip()  # noqa: E501
"""
*The Foundry Nuke* *Input Device Transform* group template.

TEMPLATE_NUKE_GROUP : str
"""

TEMPLATE_CTL_MODULE = """
// <ACEStransformID>{aces_transform_id}</ACEStransformID>
// <ACESuserName>{aces_user_name}</ACESuserName>
// Computed with {application}
// Url : {url}
// Camera Make: {camera_make}
// Camera Model: {camera_model}
// Scene adopted white : {illuminant}
// Input : Linear Camera RGB
// Output : ACES 2065-1
// Generated on : {date}

import "ACESlib.Utilities";

const float B[3][3] = {{
    {matrix}
}};

const float b[3] = {{ {multipliers} }};
const float min_b = min(b[0], min(b[1], b[2]));
const float k = {k_factor};

void main (
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut,
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn = 1.0)
{{
    // Apply exposure and white balance factors
    float Rraw = clip((b[0] * rIn * k) / min_b);
    float Graw = clip((b[1] * gIn * k) / min_b);
    float Braw = clip((b[2] * bIn * k) / min_b);

    // Apply IDT matrix
    rOut = B[0][0] * Rraw + B[0][1] * Graw + B[0][2] * Braw;
    gOut = B[1][0] * Rraw + B[1][1] * Graw + B[1][2] * Braw;
    bOut = B[2][0] * Rraw + B[2][1] * Graw + B[2][2] * Braw;
    aOut = aIn;
}}"""[1:]
"""
Color Transform Language (CTL) Module template.

TEMPLATE_CTL_MODULE : str
"""

TEMPLATE_DCTL_MODULE = """
DEFINE_ACES_PARAM(IS_PARAMETRIC_ACES_TRANSFORM: 0)

// Computed with {application}
// Url : {url}
// Camera Make: {camera_make}
// Camera Model: {camera_model}
// ACEStransformID: {aces_transform_id}
// ACESuserName: {aces_user_name}
// Scene adopted white : {illuminant}
// Input : Linear Camera RGB
// Output : ACES 2065-1
// Generated on : {date}

__CONSTANT__ float B[3][3] = {{
    {matrix}
}};

__CONSTANT__ float b[3] = {{ {multipliers} }};
__CONSTANT__ float min_b = {b_min}f;
__CONSTANT__ float k = {k_factor}f;

__DEVICE__ inline float _clipf( float v) {{
    return _fminf(v, 1.0f);
}}

__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)
{{
    // Apply exposure and white balance factors
    const float Rraw = _clipf((b[0] * p_R * k) / min_b);
    const float Graw = _clipf((b[1] * p_G * k) / min_b);
    const float Braw = _clipf((b[2] * p_B * k) / min_b);

    // Apply IDT matrix
    const float rOut = B[0][0] * Rraw + B[0][1] * Graw + B[0][2] * Braw;
    const float gOut = B[1][0] * Rraw + B[1][1] * Graw + B[1][2] * Braw;
    const float bOut = B[2][0] * Rraw + B[2][1] * Graw + B[2][2] * Braw;

    return make_float3(rOut , gOut, bOut);
}}""".strip()  # noqa: E501
"""
DaVinci Color Transform Language (DCTL) Module template.

TEMPLATE_DCTL_MODULE : str
"""


def format_float(a, decimals=10):
    """
    Format given float number at given decimal places.

    Parameters
    ----------
    a : numeric
        Float number to format.
    decimals : int, optional
        Decimal places.

    Returns
    -------
    str
        Formatted float number
    """

    return f"{{: 0.{decimals}f}}".format(a)


def format_matrix_nuke(M, decimals=10, padding=6):
    """
    Format given matrix for usage in *The Foundry Nuke*, i.e. *TCL* code for
    a *ColorMatrix* node.

    Parameters
    ----------
    M : array_like
        Matrix to format.
    decimals : int, optional
        Decimals to use when formatting the matrix.
    padding : int, optional
        Padding to use when formatting the matrix.

    Returns
    -------
    str
        *The Foundry Nuke* formatted matrix.
    """

    def pretty(V):
        """
        Prettify given vector.
        """

        return " ".join(format_float(x, decimals) for x in V)

    pad = " " * padding

    tcl = f"{{{pretty(M[0])}}}\n"
    tcl += f"{pad}{{{pretty(M[1])}}}\n"
    tcl += f"{pad}{{{pretty(M[2])}}}"

    return tcl


def format_vector_nuke(V, decimals=10):
    """
    Format given vector for usage in *The Foundry Nuke*, e.g. *TCL* code for
    a *Multiply* node.

    Parameters
    ----------
    V : array_like
        Vector to format.
    decimals : int, optional
        Decimals to use when formatting the vector.

    Returns
    -------
    str
        *The Foundry Nuke* formatted vector.
    """

    return " ".join(format_float(x, decimals) for x in V)


def format_matrix_ctl(M, decimals=10, padding=4):
    """
    Format given matrix for as *CTL* module.

    Parameters
    ----------
    M : array_like
        Matrix to format.
    decimals : int, optional
        Decimals to use when formatting the matrix.
    padding : int, optional
        Padding to use when formatting the matrix.

    Returns
    -------
    str
        *CTL* formatted matrix.
    """

    def pretty(V):
        """
        Prettify given number.
        """

        return ", ".join(format_float(x, decimals) for x in V)

    pad = " " * padding

    ctl = f"{{{pretty(M[0])} }},\n"
    ctl += f"{pad}{{{pretty(M[1])} }},\n"
    ctl += f"{pad}{{{pretty(M[2])} }}"

    return ctl


def format_vector_ctl(V, decimals=10):
    """
    Format given vector for as *CTL* module.

    Parameters
    ----------
    V : array_like
        Vector to format.
    decimals : int, optional
        Decimals to use when formatting the vector.

    Returns
    -------
    str
        *CTL* formatted vector.
    """

    return ", ".join(format_float(x, decimals) for x in V)


def format_float_dctl(a, decimals=10):
    """
    Format given float number for *DCTL* at given decimal places.

    Parameters
    ----------
    a : numeric
        Float number to format.
    decimals : int, optional
        Decimal places.

    Returns
    -------
    str
        Formatted float number
    """

    return f"{{: 0.{decimals}f}}".format(a)


def format_vector_dctl(V, decimals=10):
    """
    Format given vector as a *DCTL* module.

    Parameters
    ----------
    V : array_like
        Vector to format.
    decimals : int, optional
        Decimals to use when formatting the vector.

    Returns
    -------
    str
        *DCTL* formatted vector.
    """

    return ", ".join(format_float_dctl(x, decimals) for x in V)


def format_matrix_dctl(M, decimals=10, padding=4):
    """
    Format given matrix as a *DCTL* module.

    Parameters
    ----------
    M : array_like
        Matrix to format.
    decimals : int, optional
        Decimals to use when formatting the matrix.
    padding : int, optional
        Padding to use when formatting the matrix.

    Returns
    -------
    str
        *DCTL* formatted matrix.
    """

    def pretty(V):
        """
        Prettify given vector.
        """

        return ", ".join(format_float_dctl(x, decimals) for x in V)

    pad = " " * padding

    dctl = f"{{{pretty(M[0])} }},\n"
    dctl += f"{pad}{{{pretty(M[1])} }},\n"
    dctl += f"{pad}{{{pretty(M[2])} }}"

    return dctl


def format_idt_clf(
    aces_transform_id,
    aces_user_name,  # noqa: ARG001
    camera_make,  # noqa: ARG001
    camera_model,
    matrix,
    multipliers,
    k_factor,
    information,
):
    """
    Format the *IDT* matrix, multipliers and exposure factor :math:`k` as a
    *Common LUT Format* (CLF).

    Parameters
    ----------
    aces_transform_id : str
        *ACEStransformID* of the IDT, e.g.
        *urn:ampas:aces:transformId:v1.5:IDT.ARRI.ARRI-LogC4.a1.v1*.
    aces_user_name : str
        *ACESuserName* of the IDT, e.g. *ACES 1.0 Input - ARRI LogC4*.
    camera_make : str
        Manufacturer of the camera, e.g. *ARRI* or *RED*.
    camera_model : str
        Model of the camera, e.g. *ALEXA 35* or *V-RAPTOR XL 8K VV*.
    matrix : ArrayLike
        *IDT* matrix.
    multipliers : ArrayLike
        *IDT* multipliers.
    k_factor : float
        Exposure factor :math:`k` that results in a nominally "18% gray" object
        in the scene producing ACES values [0.18, 0.18, 0.18].
    information : dict
        Information pertaining to the *IDT* and the computation parameters.

    Returns
    -------
    str
        *CLF* file path.
    """

    root = Et.Element(
        "ProcessList",
        compCLFversion="3",
        id=aces_transform_id,
        name=f"{camera_model} to ACES2065-1",
    )

    def format_array(a):
        """Format given array :math:`a`."""

        return "\n".join(map(str, np.ravel(a).tolist()))

    et_input_descriptor = Et.SubElement(root, "InputDescriptor")
    et_input_descriptor.text = camera_model

    et_output_descriptor = Et.SubElement(root, "OutputDescriptor")
    et_output_descriptor.text = "ACES2065-1"

    et_info = Et.SubElement(root, "Info")
    et_academy_idt_calculator = Et.SubElement(et_info, "AcademyIDTCalculator")
    for key, value in information.items():
        sub_element = Et.SubElement(et_academy_idt_calculator, key)
        sub_element.text = str(value)

    root = clf_processing_elements(root, matrix, multipliers, k_factor)

    Et.indent(root)

    clf_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    clf_content += Et.tostring(root, encoding="UTF-8").decode("utf8")

    return clf_content
