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
    'TEMPLATE_DEFAULT_OUTPUT', 'TEMPLATE_NUKE_COLORMATRIX_NODE',
    'TEMPLATE_CTL_MODULE', 'nuke_format_matrix', 'ctl_format_matrix',
    'ctl_format_vector', 'slugify'
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

TEMPLATE_NUKE_COLORMATRIX_NODE = """
ColorMatrix {{
 inputs 0
 matrix {{
     {matrix}
   }}
 name "{name}"
 selected true
 xpos 0
 ypos 0
}}""" [1:]
"""
*The Foundry Nuke* *ColorMatrix* node template.

TEMPLATE_NUKE_COLORMATRIX_NODE : unicode
"""

TEMPLATE_CTL_MODULE = """
// Generated using {application}
// {path}
// Camera : {camera}
// Scene adopted white : {illuminant}
// Generated on {date}

import "utilities";

const float B[][] = {{ {matrix} }};

const float b[] = {{ {multipliers} }};
const float min_b = min(b[0], min(b[1], b[2]));
const float e_max = 1.000000;
const float k = 1.000000;

void main (
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn,
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut )
{{
    float Rraw = clip((b[0] * rIn) / (min_b * e_max));
    float Graw = clip((b[1] * gIn) / (min_b * e_max));
    float Braw = clip((b[2] * bIn) / (min_b * e_max));

    rOut = k * (B[0][0] * Rraw + B[0][1] * Graw + B[0][2] * Braw);
    gOut = k * (B[1][0] * Rraw + B[1][1] * Graw + B[1][2] * Braw);
    bOut = k * (B[2][0] * Rraw + B[2][1] * Graw + B[2][2] * Braw);
    aOut = 1.0;
}}""" [1:]
"""
Color Transform Language (CTL) Module template.

TEMPLATE_CTL_MODULE : unicode
"""


def nuke_format_matrix(M, decimals=10):
    """
    Formats given matrix for usage in *The Foundry Nuke*, i.e. *TCL* code for
    a *ColorMatrix* node.

    Parameters
    ----------
    M : array_like
        Matrix to format.
    decimals : int, optional
        Decimals to use when formatting the matrix.

    Returns
    -------
    unicode
        *The Foundry Nuke* formatted matrix.
    """

    def pretty(x):
        """
        Prettify given number.
        """

        return ' '.join(map('{{: 0.{0}f}}'.format(decimals).format, x))

    tcl = '{{{0}}}\n'.format(pretty(M[0]))
    tcl += '     {{{0}}}\n'.format(pretty(M[1]))
    tcl += '     {{{0}}}'.format(pretty(M[2]))

    return tcl


def ctl_format_matrix(M, decimals=10):
    """
    Formats given matrix for as *CTL* module.

    Parameters
    ----------
    M : array_like
        Matrix to format.
    decimals : int, optional
        Decimals to use when formatting the matrix.

    Returns
    -------
    unicode
        *CTL* formatted matrix.
    """

    def pretty(x):
        """
        Prettify given number.
        """

        return ', '.join(map('{{: 0.{0}f}}'.format(decimals).format, x))

    ctl = '{{{0}}}\n'.format(pretty(M[0]))
    ctl += '                      {{{0}}}\n'.format(pretty(M[1]))
    ctl += '                      {{{0}}}'.format(pretty(M[2]))

    return ctl


def ctl_format_vector(V, decimals=10):
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

    def pretty(x):
        """
        Prettify given number.
        """

        return ', '.join(map('{{: 0.{0}f}}'.format(decimals).format, x))

    ctl = '{{{0}}}'.format(pretty(V))

    return ctl


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
