"""
Common IDT Utilities
====================

Define the common *IDT* utilities objects.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import re
import shutil
import tempfile
import unicodedata
import xml.etree.ElementTree as Et
from functools import partial
from pathlib import Path

import colour
import colour_checker_detection
import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats
import xxhash
from colour import (
    SDS_COLOURCHECKERS,
    SDS_ILLUMINANTS,
    SpectralDistribution,
    sd_to_aces_relative_exposure_values,
)
from colour.algebra import euclidean_distance, vector_dot
from colour.characterisation import (
    optimisation_factory_Jzazbz,
    optimisation_factory_rawtoaces_v1,
    whitepoint_preserving_matrix,
)
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    List,
    NDArrayBoolean,
    NDArrayFloat,
    Sequence,
    Tuple,
)
from colour.models import RGB_COLOURSPACE_ACES2065_1, XYZ_to_IPT, XYZ_to_Oklab
from colour.utilities import as_float_array, zeros

mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "get_sds_colour_checker",
    "get_sds_illuminant",
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
    "OPTIMISATION_FACTORIES",
    "slugify",
    "list_sub_directories",
    "mask_outliers",
    "working_directory",
    "hash_file",
    "extract_archive",
    "sort_exposure_keys",
    "format_exposure_key",
    "get_clipping_threshold",
    "find_close_indices",
    "ExposureClippingFilter",
    "create_samples_macbeth_image",
    "interpolate_nan_values",
]

LOGGER = logging.getLogger(__name__)


def get_sds_colour_checker(colour_checker_name: str) -> Tuple[SpectralDistribution]:
    """
    Return the reference reflectances for the given colour checker.

    Parameters
    ----------
    colour_checker_name
        Name of the colour checker name.

    Returns
    -------
    :class:`tuple`
        Reference reflectances.
    """

    return tuple(SDS_COLOURCHECKERS[colour_checker_name].values())


def get_sds_illuminant(illuminant_name: str) -> SpectralDistribution:
    """
    Return the *ACES* reference illuminant spectral distribution, for the given
    illuminant name.

    Parameters
    ----------
    illuminant_name
        Name of the illuminant.

    Returns
    -------
    :class:`SpectralDistribution`
    """

    return SDS_ILLUMINANTS[illuminant_name]


SDS_COLORCHECKER_CLASSIC: Tuple[SpectralDistribution] = get_sds_colour_checker(
    "ISO 17321-1"
)
"""
Reference reflectances for the *ColorChecker Classic*.
"""

SD_ILLUMINANT_ACES: SpectralDistribution = get_sds_illuminant("D60")
"""
*ACES* reference illuminant spectral distribution,
i.e. ~*CIE Illuminant D Series D60*.
"""

SAMPLES_COUNT_DEFAULT: int = 24
"""
Default samples count.
"""

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC: Dict = (
    colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)
"""
Settings for the segmentation of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport* for a typical *Prosumer Camera* shoot.
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
    sds: Tuple[SpectralDistribution] = SDS_COLORCHECKER_CLASSIC,
    illuminant: SpectralDistribution = SD_ILLUMINANT_ACES,
    chromatic_adaptation_transform="CAT02",
) -> NDArrayFloat:
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
    :class:`np.ndarray`
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


RGB_COLORCHECKER_CLASSIC_ACES: NDArrayFloat = generate_reference_colour_checker()
"""
Reference *ACES* *RGB* values for the *ColorChecker Classic*.
"""


def optimisation_factory_Oklab() -> Tuple[NDArrayFloat, Callable, Callable, Callable]:
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *Oklab* colourspace.

    The objective function returns the Euclidean distance between the training
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


def optimisation_factory_IPT() -> Tuple[NDArrayFloat, Callable, Callable, Callable]:
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *IPT* colourspace.

    The objective function returns the Euclidean distance between the training
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


def error_delta_E(
    samples_test: ArrayLike, samples_reference: ArrayLike
) -> NDArrayFloat:
    """
    Compute the difference :math:`\\Delta E_{00}` between two given *RGB*
    colourspace arrays.

    Parameters
    ----------
    samples_test
        Test samples.
    samples_reference
        Reference samples.

    Returns
    -------
    :class:`np.ndarray`
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


def png_compare_colour_checkers(
    samples_test: ArrayLike, samples_reference: ArrayLike, columns: int = 6
) -> str:
    """
    Return the colour checkers comparison as *PNG* data.

    Parameters
    ----------
    samples_test
        Test samples.
    samples_reference
        Reference samples.
    columns
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
    root: Et.Element,
    matrix: ArrayLike,
    multipliers: ArrayLike,
    k_factor: float,
    use_range: bool = True,
    include_white_balance_in_clf: bool = False,
    flatten_clf: bool = True,
) -> Et.Element:
    """
    Add the *Common LUT Format* (CLF) elements for given *IDT* matrix,
    multipliers and exposure factor :math:`k` to given *XML* sub-element.

    Parameters
    ----------
    matrix
        *IDT* matrix.
    multipliers
        *IDT* multipliers.
    k_factor
        Exposure factor :math:`k` that results in a nominally "18% gray" object
        in the scene producing ACES values [0.18, 0.18, 0.18].
    use_range
        Whether to use the range node to clamp the graph before the exposure
        factor :math:`k`.
    include_white_balance_in_clf
        Whether to include the white balance multipliers in the *CLF*.
    flatten_clf
        Whether to flatten the *CLF*. into a single 1D Lut & 1 3x3 Matrix

    Returns
    -------
    :class:`Et.SubElement`
        *XML* sub-element.
    """

    def format_array(a: NDArrayFloat) -> str:
        """Format given array :math:`a` into 3 lines of 3 numbers."""
        # Reshape the array into a 3x3 matrix
        reshaped_array = a.reshape(3, 3)

        # Convert each row to a string and join them with newlines
        formatted_lines = [" ".join(map(str, row)) for row in reshaped_array]

        # Join all the lines with newline characters
        formatted_string = "\n\t\t".join(formatted_lines)

        return formatted_string

    if not flatten_clf:
        if include_white_balance_in_clf:
            et_RGB_w = Et.SubElement(
                root, "Matrix", inBitDepth="32f", outBitDepth="32f"
            )
            et_description = Et.SubElement(et_RGB_w, "Description")
            et_description.text = "White balance multipliers *b*."
            et_array = Et.SubElement(et_RGB_w, "Array", dim="3 3")
            et_array.text = f"\n\t\t{format_array(np.diag(multipliers))}"

        # TODO is this even used any more? Default param is False, and the calling
        # TODO function is hard coded to False
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

        # TODO Should this K really be written  into the clf, see the decode TODO
        et_k = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
        et_description = Et.SubElement(et_k, "Description")
        et_description.text = (
            'Exposure factor *k* that results in a nominally "18% gray" object in '
            "the scene producing ACES values [0.18, 0.18, 0.18]."
        )
        et_array = Et.SubElement(et_k, "Array", dim="3 3")
        et_array.text = f"\n\t\t{format_array(np.ravel(np.diag([k_factor] * 3)))}"

        et_M = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
        et_description = Et.SubElement(et_M, "Description")
        et_description.text = "*Input Device Transform* (IDT) matrix *B*."
        et_array = Et.SubElement(et_M, "Array", dim="3 3")
        et_array.text = f"\n\t\t{format_array(matrix)}"

    else:
        # If we are flattening the clf we output just a single matrix into the clf
        output_matrix = np.diag([k_factor] * 3) @ matrix
        if include_white_balance_in_clf:
            output_matrix = np.diag(multipliers) @ np.diag([k_factor] * 3) @ matrix

        et_M = Et.SubElement(root, "Matrix", inBitDepth="32f", outBitDepth="32f")
        et_description = Et.SubElement(et_M, "Description")
        et_description.text = "*Input Device Transform* (IDT) matrix *B*."
        et_array = Et.SubElement(et_M, "Array", dim="3 3")
        et_array.text = f"\n\t\t{format_array(output_matrix)}"

    return root


OPTIMISATION_FACTORIES: Dict = {
    "Oklab": optimisation_factory_Oklab,
    "JzAzBz": optimisation_factory_Jzazbz,
    "IPT": optimisation_factory_IPT,
    "CIE Lab": optimisation_factory_rawtoaces_v1,
}
"""
Optimisation factories.
"""


def slugify(object_: Any, allow_unicode: bool = False) -> str:
    """
    Generate a *SEO* friendly and human-readable slug from given object.

    Convert to ASCII if ``allow_unicode`` is *False*. Convert spaces or
    repeated dashes to single dashes. Remove characters that aren't
    alphanumerics, underscores, or hyphens. Convert to lowercase. Also strip
    leading and trailing whitespace, dashes, and underscores.

    Parameters
    ----------
    object_
        Object to convert to a slug.
    allow_unicode
        Whether to allow unicode characters in the generated slug.

    Returns
    -------
    :class:`str`
        Generated slug.

    References
    ----------
    -   https://github.com/django/django/blob/\
0dd29209091280ccf34e07c9468746c396b7778e/django/utils/text.py#L400

    Examples
    --------
    >>> slugify(
    ...     " Jack & Jill like numbers 1,2,3 and 4 and silly characters ?%.$!/"
    ... )
    'jack-jill-like-numbers-123-and-4-and-silly-characters'
    """

    value = str(object_)

    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    value = re.sub(r"[^\w\s-]", "", value.lower())

    return re.sub(r"[-\s]+", "-", value).strip("-_")


def list_sub_directories(
    directory: str,
    filterers: Sequence[Callable] = (
        lambda path: "__MACOSX" not in path.name,
        lambda path: path.is_dir(),
    ),
) -> List[str]:
    """
    List the sub-directories in given directory.

    Parameters
    ----------
    directory
        Directory to list the sub-directories from.
    filterers
        List of callables used to filter the sub-directories, each callable
        takes a :class:`Path` class instance as argument and returns whether to
        include or exclude the sub-directory as a bool.

    Returns
    -------
    :class:`list`
        Sub-directories in given directory.
    """

    sub_directories = [
        path
        for path in Path(directory).iterdir()
        if all(filterer(path) for filterer in filterers)
    ]

    return sub_directories


def mask_outliers(
    a: ArrayLike, axis: None | int = None, z_score: int = 3
) -> NDArrayBoolean:
    """
    Return the mask for the outliers of given array :math:`a` using the
    z-score.

    Parameters
    ----------
    a
        Array :math:`a` to return the outliers mask of.
    axis
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    z_score
        z-score threshold to mask the outliers.

    Returns
    -------
    :class:`np.ndarray`
        Mask for the outliers of given array :math:`a`.
    """

    return np.abs(scipy.stats.zscore(a, axis=axis)) > z_score


@contextlib.contextmanager
def working_directory(directory: str) -> None:
    """
    Define a context manager that temporarily sets the current working
    directory.

    Parameters
    ----------
    directory
        Current working directory to set.
    """

    current_working_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(current_working_directory)


def hash_file(path: str) -> str:
    """
    Hash the file a given path.

    Parameters
    ----------
    path
        Path to the file to hash.

    Returns
    -------
    :class:`str`
        File hash.
    """

    with open(path, "rb") as input_file:
        x = xxhash.xxh3_64()
        for chunk in iter(partial(input_file.read, 2**32), b""):
            x.update(chunk)

        return x.hexdigest()


def extract_archive(archive: str, directory: None | str = None) -> str:
    """
    Extract the archive to the given directory or a temporary directory if
    not given.

    Parameters
    ----------
    archive
        Archive to extract.
    directory
        Directory to extract the archive to.

    Returns
    -------
    :class:`str`
        Extracted directory.
    """

    if not directory:
        directory = (
            tempfile.TemporaryDirectory().name if directory is None else directory
        )

    LOGGER.info(
        'Extracting "%s" archive to "%s"...',
        archive,
        directory,
    )

    shutil.unpack_archive(archive, directory)

    return directory


def sort_exposure_keys(key: str) -> float:
    """
    Sort the data keys based on the +/- exposure.

    Parameters
    ----------
    key
        The key value for the exposure

    Returns
    -------
    :class`float`
        The sorted exposure key
    """

    # Allow for the removal of non-numeric characters while
    # keeping negative sign and decimal point
    return float(re.sub(r"[^\d.-]+", "", key))


def format_exposure_key(key: str) -> str:
    """
    Format the exposure keys for serialization, so they encompass the "+" symbol.

    Parameters
    ----------
    key
        The key value for the exposure

    Returns
    -------
    :class`str`
        The key value with or without the + symbol
    """

    # Format keys to add '+' prefix for positive keys
    key_float = float(re.sub(r"[^\d.-]+", "", key))
    if key_float > 0:
        return f"+{key_float:g}"
    else:
        return f"{key_float:g}"


def get_clipping_threshold() -> float:
    """Get the threshold for determining if RGB values are clipped. The threshold is
        2 code values in a 10 bit system

    Returns
    -------
    float
        The threshold for determining if RGB values are clipped

    """
    bit_depth = 10
    code_value_threshold = 2
    max_code_value = 2**bit_depth - 1
    threshold = float(code_value_threshold) / max_code_value
    return threshold


def find_close_indices(data: np.array, threshold: float = 0.005) -> list:
    """
    Find the indices of rows that have any values with differences below the threshold.

    Parameters
    ----------
    data : np.array
        A numpy array of shape (n, 3, 3) where each row contains RGB values.
    threshold : float, optional
        The tolerance threshold for determining if the RGB values between consecutive
        rows are similar. Default is 0.005.

    Returns
    -------
    List[int]
        A list of indices where the rows have RGB differences below the threshold.
    """
    top_indices = []
    bottom_indices = []
    n_rows = data.shape[0]

    # Search from the top
    for i in range(n_rows - 1):
        if np.any(np.abs(data[i] - data[i + 1]) < threshold):
            top_indices.append(i)
        else:
            break

    # Search from the bottom
    for i in range(n_rows - 1, 0, -1):
        if np.any(np.abs(data[i] - data[i - 1]) < threshold):
            bottom_indices.append(i)
        else:
            break

    # Combine and return the indices
    return sorted(np.array(top_indices + bottom_indices))


class ExposureClippingFilter:
    """
    The ExposureClippingFilter class is used to filter out exposure values that have
    clipped RGB values. The class compares the RGB values of each image sample in
    consecutive exposures. If any of the RGB values are less than the threshold, the
    exposure is considered clipped and should be removed.
    """

    def __init__(self, data_dict: dict, threshold: float):
        """
        Initialize the ExposureClippingFilter with a data dictionary and a threshold
        value.

        Parameters
        ----------
        data_dict : dict
            Dictionary with exposure values as keys and numpy arrays of image samples
            as values.
        threshold : float
            The threshold for the comparison.
        """
        self.data_dict = data_dict
        self.threshold = threshold

    def compare_image_samples(self, samples1: np.ndarray, samples2: np.ndarray) -> bool:
        """
        Compare the r, g, and b values of each corresponding image sample in two arrays.
        If any of the r, g, b values for any of the images are less than the threshold,
        return True. Otherwise, return False.

        Parameters
        ----------
        samples1 : np.ndarray
            First set of image samples with shape (n, 3).
        samples2 : np.ndarray
            Second set of image samples with shape (n, 3).

        Returns
        -------
        bool
            True if any r, g, b values are less than the threshold, False otherwise.
        """
        if not isinstance(samples1, np.ndarray):
            samples1 = np.array(samples1)

        if not isinstance(samples2, np.ndarray):
            samples2 = np.array(samples2)

        difference = np.abs(samples1 - samples2)
        return np.any(difference < self.threshold)

    def filter_samples(self) -> list:
        """
        March through exposure values from lowest to highest and from highest to lowest,
        comparing the samples. If any of the comparisons fail, stores the exposure
        value. Stops when an exposure passes the test, and returns all failed exposure
        values.

        Returns
        -------
        list
            A list of exposure values that failed the threshold test and should be
            removed
        """
        exposure_values = sorted(self.data_dict.keys())
        failed_exposures = []

        # March from the bottom (smallest exposure to largest)
        for i in range(len(exposure_values) - 1):
            current_exposure = exposure_values[i]
            next_exposure = exposure_values[i + 1]

            if self.compare_image_samples(
                self.data_dict[current_exposure], self.data_dict[next_exposure]
            ):
                failed_exposures.append(current_exposure)
            else:
                break

        # March from the top (largest exposure to smallest)
        for i in range(len(exposure_values) - 1, 0, -1):
            current_exposure = exposure_values[i]
            previous_exposure = exposure_values[i - 1]

            if self.compare_image_samples(
                self.data_dict[current_exposure], self.data_dict[previous_exposure]
            ):
                failed_exposures.append(current_exposure)
            else:
                break

        # TODO we should update the algo above so that we march from the bottom to 0.0
        # TODO and from the top down to 0.0, vs just doing all of them as we want
        #  0.0 to always remain. This hack can go
        result = sorted(set(failed_exposures))
        if 0.0 in result:
            result.remove(0.0)

        return result


def create_samples_macbeth_image(
    colours: np.array,
    reduction_percent: int = 30,
    colours_per_row: int = 6,
    target_width: int = 1920,
    target_height: int = 1080,
) -> np.array:
    """
    Create a numpy image from a list of colours and creates a macbeth chart "like" image
    with the colours provided.

    A boarder is left around the edge of the image to allow for the reduction in size,
    and set to be the same colour as the -3 index of the colours array. aka the grey of
    the macbeth chart.

    The image is not a macbeth chart, as the colours are as read from the camera samples
    vs those of an actual macbeth chart.

    Parameters
    ----------
    colours: np.array
        An array of colours in [R, G, B] format

    reduction_percent: int
        The percentage to reduce the target width and height by to leave a boarder
        around the edge

    colours_per_row: int
        The number of colours to put in each row

    target_width: int
        The target width of the image

    target_height: int
        The target height of the image

    Returns
    -------
    np.array
        The image as a numpy array

    """
    num_rows = (
        len(colours) + colours_per_row - 1
    ) // colours_per_row  # Calculate the number of rows needed

    # Calculate the new dimensions with reduction percentage
    reduced_width = int(target_width * (1 - reduction_percent / 100))
    reduced_height = int(target_height * (1 - reduction_percent / 100))

    strip_width = int(reduced_width / colours_per_row)
    strip_height = int(reduced_height / num_rows)

    # Calculate the starting point to center the strips in the original target
    # dimensions
    start_x = (target_width - reduced_width) // 2
    start_y = (target_height - reduced_height) // 2

    # Create an empty image with the background color from the -3 index of the colour
    # array
    background_color = colours[-3]  # Assuming colours[-3] is in [R, G, B] format
    image = np.full(
        (target_height, target_width, 3), background_color, dtype=np.float32
    )

    # Fill the image with the color strips
    for i, color in enumerate(colours):
        row = i // colours_per_row
        col = i % colours_per_row
        x = start_x + col * strip_width
        y = start_y + row * strip_height
        image[y : y + strip_height, x : x + strip_width] = color

    return image


def interpolate_nan_values(array: np.array) -> np.array:
    """Interpolate the NaN values in a 2D array using linear interpolation.

    Parameters
    ----------
    array: np.array
        The 2D array to interpolate

    Returns
    -------
    np.array
        The array with the NaN values interpolated
    """

    # Convert the 2D array to a DataFrame for easy interpolation
    df = pd.DataFrame(array)

    # Interpolate the entire DataFrame
    # Forward fill and backward fill remaining NaNs (at the edges)
    interpolated_df = df.fillna(method="ffill").fillna(method="bfill")

    interpolated_df = interpolated_df.interpolate(method="linear", axis=0)

    # Replace only the NaN values in the original array with interpolated values
    final_array = np.where(np.isnan(array), interpolated_df.to_numpy(), array)

    return final_array
