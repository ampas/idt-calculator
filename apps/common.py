# -*- coding: utf-8 -*-
"""
Common
======
"""

import colour
import colour_datasets
import re

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2018-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'COLOUR_ENVIRONMENT', 'DATASET_RAW_TO_ACES',
    'TRAINING_DATA_KODAK190PATCHES', 'MSDS_CAMERA_SENSITIVITIES',
    'CAMERA_SENSITIVITIES_OPTIONS', 'CAT_OPTIONS', 'ILLUMINANT_OPTIONS',
    'TEMPLATE_DEFAULT_OUTPUT', 'TEMPLATE_NUKE_GROUP', 'TEMPLATE_CTL_MODULE',
    'format_float', 'format_matrix_nuke', 'format_vector_nuke',
    'format_matrix_ctl', 'format_vector_ctl', 'slugify'
]

COLOUR_ENVIRONMENT = None
"""
*Colour* environment formatted as a string.

COLOUR_ENVIRONMENT : unicode
"""


def _print_colour_environment(describe):
    """
    Intercepts colour environment description output to save it as a formatted
    strings.
    """

    global COLOUR_ENVIRONMENT

    if COLOUR_ENVIRONMENT is None:
        COLOUR_ENVIRONMENT = ''

    print(describe)

    COLOUR_ENVIRONMENT += describe
    COLOUR_ENVIRONMENT += '\n'


colour.utilities.describe_environment(print_callable=_print_colour_environment)

DATASET_RAW_TO_ACES = colour_datasets.load(
    'RAW to ACES Utility Data - Dyer et al. (2017)')

TRAINING_DATA_KODAK190PATCHES = DATASET_RAW_TO_ACES['training']['190-patch']
"""
*Kodak* 190 patches training data.

TRAINING_DATA_KODAK190PATCHES : MultiSpectralDistributions
"""

MSDS_CAMERA_SENSITIVITIES = DATASET_RAW_TO_ACES['camera']

CAMERA_SENSITIVITIES_OPTIONS = [{
    'label': key,
    'value': key
} for key in sorted(MSDS_CAMERA_SENSITIVITIES)]
"""
Camera sensitivities options for a :class:`Dropdown` class instance.

CAMERA_SENSITIVITIES_OPTIONS : list
"""
CAMERA_SENSITIVITIES_OPTIONS.insert(0, {'label': 'Custom', 'value': 'Custom'})

CAT_OPTIONS = [{
    'label': key,
    'value': key
} for key in sorted(colour.CHROMATIC_ADAPTATION_TRANSFORMS.keys())]
"""
*Chromatic adaptation transform* options for a :class:`Dropdown` class
instance.

CAT_OPTIONS : list
"""

ILLUMINANT_OPTIONS = [{
    'label': key,
    'value': key
} for key in sorted(colour.SDS_ILLUMINANTS.keys())]
"""
Illuminant options for a :class:`Dropdown`class instance.

ILLUMINANTS_OPTIONS : list
"""
ILLUMINANT_OPTIONS.insert(0, {'label': 'Custom', 'value': 'Custom'})

TEMPLATE_DEFAULT_OUTPUT = """
IDT Matrix
----------

{0}

White Balance Multipliers
-------------------------

{1}""" [1:]
"""
Default formatting template.

TEMPLATE_DEFAULT_OUTPUT : unicode
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
""" [1:]
"""
*The Foundry Nuke* *Input Device Transform* group template.

TEMPLATE_NUKE_GROUP : unicode
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
}}""" [1:]
"""
Color Transform Language (CTL) Module template.

TEMPLATE_CTL_MODULE : unicode
"""


def format_float(a, decimals=10):
    """
    Formats given float number at given decimal places.

    Parameters
    ----------
    a : numeric
        Float number to format.
    decimals : int, optional
        Decimal places.

    Returns
    -------
    unicode
        Formatted float number
    """

    return f'{{: 0.{decimals}f}}'.format(a)


def format_matrix_nuke(M, decimals=10, padding=6):
    """
    Formats given matrix for usage in *The Foundry Nuke*, i.e. *TCL* code for
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
    unicode
        *The Foundry Nuke* formatted matrix.
    """

    def pretty(x):
        """
        Prettify given number.
        """

        return ' '.join(map(lambda x: format_float(x, decimals), x))

    pad = ' ' * padding

    tcl = f'{{{pretty(M[0])}}}\n'
    tcl += f'{pad}{{{pretty(M[1])}}}\n'
    tcl += f'{pad}{{{pretty(M[2])}}}'

    return tcl


def format_vector_nuke(V, decimals=10):
    """
    Formats given vector for usage in *The Foundry Nuke*, e.g. *TCL* code for
    a *Multiply* node.

    Parameters
    ----------
    V : array_like
        Vector to format.
    decimals : int, optional
        Decimals to use when formatting the vector.

    Returns
    -------
    unicode
        *The Foundry Nuke* formatted vector.
    """

    return ' '.join(map(lambda x: format_float(x, decimals), V))


def format_matrix_ctl(M, decimals=10, padding=4):
    """
    Formats given matrix for as *CTL* module.

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
    unicode
        *CTL* formatted matrix.
    """

    def pretty(x):
        """
        Prettify given number.
        """

        return ', '.join(map(lambda x: format_float(x, decimals), x))

    pad = ' ' * padding

    ctl = f'{{{pretty(M[0])} }},\n'
    ctl += f'{pad}{{{pretty(M[1])} }},\n'
    ctl += f'{pad}{{{pretty(M[2])} }}'

    return ctl


def format_vector_ctl(V, decimals=10):
    """
    Formats given vector for as *CTL* module.

    Parameters
    ----------
    V : array_like
        Vector to format.
    decimals : int, optional
        Decimals to use when formatting the vector.

    Returns
    -------
    unicode
        *CTL* formatted vector.
    """

    return ', '.join(map(lambda x: format_float(x, decimals), V))


def slugify(a):
    """
    Slugifies given string to remove non-programmatic friendly characters.

    Parameters
    ----------
    a : unicode
        String to slugify.

    Returns
    -------
    unicode
        Slugified string.
    """

    return re.sub(r'\s|-|\.', '_',
                  re.sub(r'(?u)[^-\w.]', ' ',
                         str(a).strip()).strip().lower())
