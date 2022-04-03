"""
Common IDT Utilities
====================
"""

import base64
import colour
import io
import matplotlib
import numpy as np
import re

from colour.algebra import euclidean_distance, vector_dot
from colour.models import RGB_COLOURSPACE_ACES2065_1, XYZ_to_Oklab, XYZ_to_IPT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "slugify",
    "error_delta_E",
    "png_compare_colour_checkers",
    "optimisation_factory_Oklab",
    "optimisation_factory_IPT",
]


def slugify(a):
    """
    Slugify given string to remove non-programmatic friendly characters.

    Parameters
    ----------
    a : str
        String to slugify.

    Returns
    -------
    str
        Slugified string.
    """

    return re.sub(
        r"\s|-|\.",
        "_",
        re.sub(r"(?u)[^-\w.]", " ", str(a).strip()).strip(),
    )


def error_delta_E(samples_test, samples_reference):
    """
    Compute the difference :math:`\\Delta E_{00}` between two given *RGB*
    colourspace arrays.

    Parameters
    ----------
    samples_test : array_like
        Test samples.
    samples_reference : array_like
        Reference samples.

    Returns
    -------
    NDArray
        :math:`\\Delta E_{00}`.
    """

    XYZ_to_RGB_kargs = {
        "illuminant_XYZ": RGB_COLOURSPACE_ACES2065_1.whitepoint,
        "illuminant_RGB": RGB_COLOURSPACE_ACES2065_1.whitepoint,
        "matrix_XYZ_to_RGB": RGB_COLOURSPACE_ACES2065_1.matrix_XYZ_to_RGB,
    }

    Lab_test = (
        colour.convert(
            samples_test, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs
        )
        * 100
    )
    Lab_reference = (
        colour.convert(
            samples_reference, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs
        )
        * 100
    )

    return colour.delta_E(Lab_test, Lab_reference)


def png_compare_colour_checkers(samples_test, samples_reference, columns=6):
    """
    Return the colour checkers comparison as *PNG* data.

    Parameters
    ----------
    samples_test : array_like
        Test samples.
    samples_reference : array_like
        Reference samples.
    columns : integer, optional
        Number of columns for the colour checkers comparison.

    Returns
    -------
    str
        *PNG* data.
    """

    colour.plotting.plot_multi_colour_swatches(
        list(zip(samples_reference, samples_test)),
        columns=columns,
        compare_swatches="Stacked",
        direction="-y",
    )
    colour.plotting.render(
        **{
            "standalone": False,
        }
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png


def optimisation_factory_Oklab():
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *Oklab* colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the *Oklab* colourspace.

    Returns
    -------
    :class:`tuple`
        Objective function and *CIE XYZ* colourspace to *Oklab* colourspace
        function.

    Examples
    --------
    >>> optimisation_factory_Oklab()  # doctest: +SKIP
    (<function optimisation_factory_Oklab.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Oklab.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(M, RGB, Jab):
        """*Oklab* colourspace based objective function."""

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_Oklab(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *Oklab* colourspace function."""

        return XYZ_to_Oklab(XYZ)

    return objective_function, XYZ_to_optimization_colour_model


def optimisation_factory_IPT():
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *IPT* colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the *IPT* colourspace.

    Returns
    -------
    :class:`tuple`
        Objective function and *CIE XYZ* colourspace to *IPT* colourspace
        function.

    Examples
    --------
    >>> optimisation_factory_IPT()  # doctest: +SKIP
    (<function optimisation_factory_IPT.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_IPT.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(M, RGB, Jab):
        """*IPT* colourspace based objective function."""

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_IPT(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *IPT* colourspace function."""

        return XYZ_to_IPT(XYZ)

    return objective_function, XYZ_to_optimization_colour_model
