"""
IDT Color Space Generator
=============================

Define the *IDT* generator class for a known colour space and encoding
"""

import logging

import colour
import matplotlib as mpl
import numpy as np
from colour import LUT3x1D
from colour.hints import NDArrayFloat
from colour.models.rgb import transfer_functions

from aces.idt.generators.prosumer_camera import IDTGeneratorProsumerCamera

# TODO are the mpl.use things needed in every file?
mpl.use("Agg")

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTGeneratorColourSpace",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorColourSpace(IDTGeneratorProsumerCamera):
    """
    Define an *IDT* generator for a *Known Colour Space & Transform*.

    Parameters
    ----------
    project_settings : IDTProjectSettings, optional
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTBaseGenerator.GENERATOR_NAME`

    Methods
    -------
    -   :meth:`~aces.idt.IDTBaseGenerator.generate_LUT`
    -   :meth:`~aces.idt.IDTBaseGenerator.filter_LUT`
    -   :meth:`~aces.idt.IDTBaseGenerator.decode`
    -   :meth:`~aces.idt.IDTBaseGenerator.optimise`
    """

    GENERATOR_NAME = "IDTGeneratorColourSpace"
    """*IDT* generator name."""

    def __init__(self, project_settings):
        super().__init__(project_settings)

    def sample(self):
        """
        We do not sample anything as we are not working from footage. We simply override

        Returns
        -------
            None
        """

    def sort(self, start_index: int = -6) -> np.ndarray:  # noqa: ARG002
        """We do not sort anything as we are not working from footage.

        Parameters
        ----------
        start_index : int, optional
            Start index for sorting, by default -6

        Returns
        -------
        :class:`np.ndarray`
            Empty array
        """
        return np.array([])

    def remove_clipping(self) -> np.ndarray:
        """We do not remove clipping as we are not working from footage.

        Returns
        -------
        :class:`np.ndarray`
            Empty array

        """
        return np.array([])

    def generate_LUT(self) -> LUT3x1D:
        """We do not do any calculations to generate the LUT3x1D.

        We build an identity LUT and apply either the known transfer function or a
        provided clf.

        Returns
        -------
        :class:`LUT3x1D`
            The lut generated from a known transform function or a provided clf.
        """
        size = self.project_settings.lut_size
        LOGGER.info('Generating unfiltered "LUT3x1D" with "%s" size...', size)

        self._LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")

        tf = transfer_functions.LOG_DECODINGS[
            self.project_settings.encoding_transfer_function
        ]
        self._LUT_unfiltered.table = tf(self._LUT_unfiltered.table)

        return self._LUT_unfiltered

    def filter_LUT(self) -> LUT3x1D:
        """As the lut was generated vs calculated we do not filter the LUT;
        we simply copy it.

        Returns
        -------
        :class:`LUT3x1D`
            Copy of the unfiltered LUT
        """
        LOGGER.info('No Filtering Simply Copying "LUT3x1D"')
        self._LUT_filtered = self._LUT_unfiltered.copy()
        self._LUT_filtered.name = "LUT - Filtered"
        return self._LUT_filtered

    def decode(self) -> None:
        """We do not do any calculations we simple copy the filtered
        LUT to the decoding LUT.

        """
        self._LUT_decoding = self._LUT_filtered.copy()

    def optimise(self) -> tuple[NDArrayFloat]:
        """As we are not working from footage here we do not need to optimise anything
        we simply set the k factor and RGB_w to identity values.

        We calculate the M matrix from the encoding colourspace

        Returns
        -------
        :class:`tuple`
            Tuple of *IDT* matrix :math:`M`, white balance multipliers
            :math:`RGB_w` and exposure factor :math:`k`
        """
        encoding_colourspace = colour.RGB_COLOURSPACES[
            self.project_settings.encoding_colourspace
        ]
        self.calculate_M_from_colourspace(encoding_colourspace)
        self._RGB_w = np.array([1.0, 1.0, 1.0])
        self._k = 1.0

        return self._M, self._RGB_w, self._k

    def calculate_M_from_colourspace(self, encoding_colourspace) -> None:
        """Calculate the M matrix from the encoding colourspace provided and
            ACES2065-1, and store it

        Parameters
        ----------
        encoding_colourspace : RGB_Colourspace
            The encoding colourspace to calculate the M matrix from

        """
        target_colourspace = colour.RGB_COLOURSPACES["ACES2065-1"]
        transformation_matrix = colour.matrix_RGB_to_RGB(
            encoding_colourspace,
            target_colourspace,
            chromatic_adaptation_transform=self.project_settings.cat,
        )
        self._M = transformation_matrix
