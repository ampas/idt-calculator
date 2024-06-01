"""
Common IDT Utilities
====================
"""

import base64
import io
import re
import xml.etree.ElementTree as Et

import colour
import colour_checker_detection
import cv2
import matplotlib as mpl
import numpy as np
from colour import (
    SDS_COLOURCHECKERS,
    SDS_ILLUMINANTS,
    sd_to_aces_relative_exposure_values,
)
from colour.algebra import euclidean_distance, vector_dot
from colour.characterisation import (
    optimisation_factory_Jzazbz,
    optimisation_factory_rawtoaces_v1,
    whitepoint_preserving_matrix,
)
from colour.models import RGB_COLOURSPACE_ACES2065_1, XYZ_to_IPT, XYZ_to_Oklab
from colour.utilities import as_float_array, zeros

mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "get_sds_colour_checker",
    "get_sds_illuminant",
    "OPTIMISATION_FACTORIES",
    "SDS_COLORCHECKER_CLASSIC",
    "SD_ILLUMINANT_ACES",
    "SAMPLES_COUNT_DEFAULT",
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "generate_reference_colour_checker",
    "RGB_COLORCHECKER_CLASSIC_ACES",
    "optimisation_factory_Oklab",
    "optimisation_factory_IPT",
    "error_delta_E",
    "png_compare_colour_checkers",
    "clf_processing_elements",
]


def get_sds_colour_checker(colour_checker_name):
    """
    Get the Reference reflectances for the given colour checker

    Parameters
    ----------
    colour_checker_name: The name of the colour checker name

    Returns
    -------
        SDS_COLORCHECKER : tuple

    """
    return tuple(SDS_COLOURCHECKERS[colour_checker_name].values())


def get_sds_illuminant(illuminant_name):
    """
    Get *ACES* reference illuminant spectral distribution, for the given illuminant name

    Parameters
    ----------
    illuminant_name: The name of the illuminant

    Returns
    -------
        SpectralDistribution

    """
    return SDS_ILLUMINANTS[illuminant_name]


SDS_COLORCHECKER_CLASSIC = get_sds_colour_checker("ISO 17321-1")
"""
Reference reflectances for the *ColorChecker Classic*.

SDS_COLORCHECKER_CLASSIC : tuple
"""

SD_ILLUMINANT_ACES = get_sds_illuminant("D60")
"""
*ACES* reference illuminant spectral distribution,
i.e. ~*CIE Illuminant D Series D60*.

SD_ILLUMINANT_ACES : SpectralDistribution
"""

SAMPLES_COUNT_DEFAULT = 24
"""
Default samples count.

SAMPLES_COUNT_DEFAULT : int
"""

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC = (
    colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)
"""
Settings for the segmentation of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport* for a typical *Prosumer Camera* shoot.

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC : dict
"""

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
    {
        "working_width": 1600,
        "working_height": int(1600 * 4 / 6),
        "adaptive_threshold_kwargs": {
            "maxValue": 255,
            "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
            "thresholdType": cv2.THRESH_BINARY,
            "blockSize": int(1600 * 0.015) - int(1600 * 0.015) % 2 + 1,
            "C": 2,
        },
    }
)


def generate_reference_colour_checker(
    sds=SDS_COLORCHECKER_CLASSIC,
    illuminant=SD_ILLUMINANT_ACES,
    chromatic_adaptation_transform="CAT02",
):
    """
    Generate the reference *ACES* *RGB* values for the *ColorChecker Classic*.

    Parameters
    ----------
    sds : tuple, optional
        *ColorChecker Classic* reflectances.
    illuminant : SpectralDistribution, optional
        Spectral distribution of the illuminant to compute the reference
        *ACES* *RGB* values.
    chromatic_adaptation_transform : str
        *Chromatic adaptation* transform.

    Returns
    -------
    NDArray
        Reference *ACES* *RGB* values.
    """

    return as_float_array(
        [
            sd_to_aces_relative_exposure_values(
                sd,
                illuminant,
                chromatic_adaptation_transform=chromatic_adaptation_transform,
            )
            for sd in sds
        ]
    )


RGB_COLORCHECKER_CLASSIC_ACES = generate_reference_colour_checker()
"""
Reference *ACES* *RGB* values for the *ColorChecker Classic*.

RGB_COLORCHECKER_CLASSIC_ACES : NDArray
"""


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
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *Oklab* colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_Oklab()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_Oklab.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Oklab.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_Oklab.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(M, RGB, Jab):
        """*Oklab* colourspace based objective function."""

        M = whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_Oklab(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *Oklab* colourspace function."""

        return XYZ_to_Oklab(XYZ)

    def finaliser_function(M):
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


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
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *IPT* colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_IPT()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_IPT.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_IPT.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_IPT.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(M, RGB, Jab):
        """*IPT* colourspace based objective function."""

        M = whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_IPT(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *IPT* colourspace function."""

        return XYZ_to_IPT(XYZ)

    def finaliser_function(M):
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
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
        colour.convert(samples_test, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs)
        * 100
    )
    Lab_reference = (
        colour.convert(samples_reference, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs)
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
            "show": False,
        }
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png


def clf_processing_elements(
    root,
    matrix,
    multipliers,
    k_factor,
    use_range=True,
):
    """
    Add the *Common LUT Format* (CLF) elements for given *IDT* matrix,
    multipliers and exposure factor :math:`k` to given *XML* sub-element.

    Parameters
    ----------
    matrix : ArrayLike
        *IDT* matrix.
    multipliers : ArrayLike
        *IDT* multipliers.
    k_factor : float
        Exposure factor :math:`k` that results in a nominally "18% gray" object
        in the scene producing ACES values [0.18, 0.18, 0.18].
    use_range : bool
        Whether to use the range node to clamp the graph before the exposure
        factor :math:`k`.

    Returns
    -------
    Et.SubElement
        *XML* sub-element.
    """

    def format_array(a):
        """Format given array :math:`a`."""

        return re.sub(r"\[|\]|,", "", "\n".join(map(str, a.tolist())))

    et_RGB_w = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
    et_description = Et.SubElement(et_RGB_w, "Description")
    et_description.text = "White balance multipliers *b*."
    et_array = Et.SubElement(et_RGB_w, "Array", dim="3 3")
    et_array.text = f"\n{format_array(np.diag(multipliers))}"

    if use_range:
        et_range = Et.SubElement(
            root,
            "Range",
            inBitDepth="32f",
            outBitDepth="32f",
        )
        et_max_in_value = Et.SubElement(et_range, "maxInValue")
        et_max_in_value.text = "1"
        et_max_out_value = Et.SubElement(et_range, "maxOutValue")
        et_max_out_value.text = "1"

    et_k = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
    et_description = Et.SubElement(et_k, "Description")
    et_description.text = (
        'Exposure factor *k* that results in a nominally "18% gray" object in '
        "the scene producing ACES values [0.18, 0.18, 0.18]."
    )
    et_array = Et.SubElement(et_k, "Array", dim="3 3")
    et_array.text = f"\n{format_array(np.ravel(np.diag([k_factor] * 3)))}"

    et_M = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
    et_description = Et.SubElement(et_M, "Description")
    et_description.text = "*Input Device Transform* (IDT) matrix *B*."
    et_array = Et.SubElement(et_M, "Array", dim="3 3")
    et_array.text = f"\n{format_array(matrix)}"

    return root


OPTIMISATION_FACTORIES = {
    "Oklab": optimisation_factory_Oklab,
    "JzAzBz": optimisation_factory_Jzazbz,
    "IPT": optimisation_factory_IPT,
    "CIE Lab": optimisation_factory_rawtoaces_v1,
}
"""
Optimisation factories.

OPTIMISATION_FACTORIES : dict
"""
