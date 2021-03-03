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
    'CAMERA_SENSITIVITIES_OPTIONS', 'CAT_OPTIONS',
    'ILLUMINANT_OPTIONS', 'NUKE_COLORMATRIX_NODE_TEMPLATE',
    'nuke_format_matrix', 'slugify'
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

NUKE_COLORMATRIX_NODE_TEMPLATE = """
ColorMatrix {{
 inputs 0
 matrix {{
     {0}
   }}
 name "{1}"
 selected true
 xpos 0
 ypos 0
}}""" [1:]
"""
*The Foundry Nuke* *ColorMatrix* node template.

NUKE_COLORMATRIX_NODE_TEMPLATE : unicode
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
