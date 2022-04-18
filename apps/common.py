"""
Common Apps Utilities
=====================
"""

import colour
import colour_checker_detection  # noqa
import colour_datasets
from colour import (
    CubicSplineInterpolator,
    LinearInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
)
from colour.characterisation import (
    optimisation_factory_rawtoaces_v1,
    optimisation_factory_Jzazbz,
)

from aces.idt import (
    optimisation_factory_Oklab,
    optimisation_factory_IPT,
)


__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"


__all__ = [
    "COLOUR_ENVIRONMENT",
    "STYLE_DATATABLE",
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
    "OPTIMISATION_FACTORIES",
    "OPTIONS_DISPLAY_COLOURSPACES",
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
]

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

    global COLOUR_ENVIRONMENT

    if COLOUR_ENVIRONMENT is None:
        COLOUR_ENVIRONMENT = ""

    print(describe)

    COLOUR_ENVIRONMENT += describe
    COLOUR_ENVIRONMENT += "\n"


colour.utilities.describe_environment(print_callable=_print_colour_environment)

STYLE_DATATABLE = {
    "header_background_colour": "rgb(30, 30, 30)",
    "header_colour": "rgb(220, 220, 220)",
    "cell_background_colour": "rgb(50, 50, 50)",
    "cell_colour": "rgb(220, 220, 220)",
}
"""
Datatable stylesheet.

STYLE_DATATABLE : dict
"""

DATATABLE_DECIMALS = 7
"""
Datatable decimals.

DATATABLE_DECIMALS : int
"""

CUSTOM_WAVELENGTHS = list(range(380, 395, 5)) + ["..."]
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
    {"label": key, "value": key}
    for key in sorted(colour.SDS_ILLUMINANTS.keys())
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

OPTIMISATION_FACTORIES = {
    "Oklab": optimisation_factory_Oklab,
    "IPT": optimisation_factory_IPT,
    "JzAzBz": optimisation_factory_Jzazbz,
    "CIE Lab": optimisation_factory_rawtoaces_v1,
}
"""
Optimisation factories.

OPTIMISATION_FACTORIES : dict
"""

OPTIONS_DISPLAY_COLOURSPACES = [
    {"label": key, "value": key} for key in ["sRGB", "Display P3"]
]
"""
Display colourspaces.

OPTIONS_DISPLAY_COLOURSPACES : list
"""


OPTIONS_OPTIMISATION_SPACES = [
    {"label": key, "value": key} for key in OPTIMISATION_FACTORIES.keys()
]
"""
Optimisation colourspaces.

OPTIONS_OPTIMISATION_SPACES : list
"""


TEMPLATE_DEFAULT_OUTPUT = """
IDT Matrix
----------

{0}

White Balance Multipliers
-------------------------

{1}"""[
    1:
]
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
\nCamera : {camera}\
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
""".strip()  # noqa
"""
*The Foundry Nuke* *Input Device Transform* group template.

TEMPLATE_NUKE_GROUP : str
"""

TEMPLATE_CTL_MODULE = """
// Computed with {application}
// Url : {url}
// Camera : {camera}
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
}}"""[
    1:
]
"""
Color Transform Language (CTL) Module template.

TEMPLATE_CTL_MODULE : str
"""

TEMPLATE_DCTL_MODULE = """
DEFINE_ACES_PARAM(IS_PARAMETRIC_ACES_TRANSFORM: 0)

// Computed with {application}
// Url : {url}
// Camera : {camera}
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
}}""".strip()  # noqa
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

    def pretty(x):
        """
        Prettify given number.
        """

        return " ".join(map(lambda x: format_float(x, decimals), x))

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

    return " ".join(map(lambda x: format_float(x, decimals), V))


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

    def pretty(x):
        """
        Prettify given number.
        """

        return ", ".join(map(lambda x: format_float(x, decimals), x))

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

    return ", ".join(map(lambda x: format_float(x, decimals), V))


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
    Format given vector for as *DCTL* module.

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

    return ", ".join(map(lambda x: format_float_dctl(x, decimals), V))


def format_matrix_dctl(M, decimals=10, padding=4):
    """
    Format given matrix for as *DCTL* module.

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

    def pretty(x):
        """
        Prettify given number.
        """

        return ", ".join(map(lambda x: format_float_dctl(x, decimals), x))

    pad = " " * padding

    dctl = f"{{{pretty(M[0])} }},\n"
    dctl += f"{pad}{{{pretty(M[1])} }},\n"
    dctl += f"{pad}{{{pretty(M[2])} }}"

    return dctl
