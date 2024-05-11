"""
Module holds a common base generator class for which all other IDT generators inherit
from
"""
import base64
import io
import logging
import os
import re
import shutil
import xml.etree.ElementTree as Et
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import colour
import cv2
import jsonpickle
import numpy as np
from colour import LUT1D, read_image
from colour.utilities import Structure, as_float_array, zeros
from colour_checker_detection.detection import (
    as_int32_array,
    reformat_image,
    sample_colour_checker,
    segmenter_default,
)
from matplotlib import pyplot as plt

from aces.idt import clf_processing_elements
from aces.idt.core import common
from aces.idt.core.constants import DataFolderStructure
from aces.idt.core.utilities import mask_outliers, working_directory

logger = logging.getLogger(__name__)


class IDTBaseGenerator(ABC):
    """
    A Class which holds the core logic for any generator, and should be inherited from
    """

    def __init__(self, project_settings):
        self._project_settings = project_settings
        self._samples_analysis = None

        self._samples_camera = None
        self._samples_reference = None
        self._baseline_exposure = 0

        self._LUT_unfiltered = None
        self._LUT_filtered = None
        self._LUT_decoding = None

        self._M = None
        self._RGB_w = None
        self._k = None

        self._image_grey_card_sampling = None
        self._image_colour_checker_segmentation = None

    @property
    def project_settings(self):
        """Returns the project settings passed into the generator from the application

        Returns
        -------
        IDTProjectSettings
            The ProjectSettings passed into the generator
        """
        return self._project_settings

    @property
    def image_colour_checker_segmentation(self):
        """
        Getter property for the image of the colour checker with segmentation
        contours.

        Returns
        -------
        :class:`NDArray` or None
            Image of the colour checker with segmentation contours.
        """

        return self._image_colour_checker_segmentation

    @property
    def baseline_exposure(self):
        """
        Getter property for the baseline exposure.

        Returns
        -------
        :class:`float`
            Baseline exposure.
        """

        return self._baseline_exposure

    @property
    def image_grey_card_sampling(self):
        """
        Getter property for the image the grey card with sampling contours.
        contours.

        Returns
        -------
        :class:`NDArray` or None
            Image of the grey card with sampling contours.
        """

        return self._image_grey_card_sampling

    @property
    def samples_camera(self):
        """
        Getter property for the samples of the camera produced by the sorting
        process.

        Returns
        -------
        :class:`NDArray` or None
            Samples of the camera produced by the sorting process.
        """

        return self._samples_camera

    @property
    def samples_reference(self):
        """
        Getter property for the reference samples produced by the sorting
        process.

        Returns
        -------
        :class:`NDArray` or None
            Reference samples produced by the sorting process.
        """

        return self._samples_reference

    @property
    def LUT_unfiltered(self):
        """
        Getter property for the unfiltered *LUT*.

        Returns
        -------
        :class:`LUT3x1D` or None
            Unfiltered *LUT*.
        """

        return self._LUT_unfiltered

    @property
    def LUT_filtered(self):
        """
        Getter property for the filtered *LUT*.

        Returns
        -------
        :class:`LUT3x1D` or None
            Filtered *LUT*.
        """

        return self._LUT_filtered

    @property
    def LUT_decoding(self):
        """
        Getter property for the (final) decoding *LUT*.

        Returns
        -------
        :class:`LUT1D` or :class:`LUT3x1D` or None
            Decoding *LUT*.
        """

        return self._LUT_decoding

    @property
    def M(self):
        """
        Getter property for the *IDT* matrix :math:`M`,

        Returns
        -------
        :class:`NDArray` or None
           *IDT* matrix :math:`M`.
        """

        return self._M

    @property
    def RGB_w(self):
        """
        Getter property for the white balance multipliers :math:`RGB_w`.

        Returns
        -------
        :class:`NDArray` or None
            White balance multipliers :math:`RGB_w`.
        """

        return self._RGB_w

    @property
    def k(self):
        """
        Getter property for the exposure factor :math:`k` that results in a
        nominally "18% gray" object in the scene producing ACES values
        [0.18, 0.18, 0.18].

        Returns
        -------
        :class:`NDArray` or None
            Exposure factor :math:`k`
        """

        return self._k

    @property
    def samples_analysis(self):
        """
        Getter property for the samples produced by the colour checker sampling
        process.

        Returns
        -------
        :class:`NDArray` or None
            Samples produced by the colour checker sampling process.
        """

        return self._samples_analysis

    def sample(self):
        """
        Sample the images from the *IDT* specification.
        """

        logger.info('Sampling "IDT" specification images...')

        settings = Structure(**common.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
        working_width = settings.working_width
        working_height = settings.working_height

        def _reformat_image(image):
            """Reformat given image."""

            return reformat_image(
                image, settings.working_width, settings.interpolation_method
            )

        rectangle = as_int32_array(
            [
                [working_width, 0],
                [working_width, working_height],
                [0, working_height],
                [0, 0],
            ]
        )

        self._samples_analysis = deepcopy(self.project_settings.data)
        # Baseline exposure value, it can be different from zero.
        if 0 not in self.project_settings.data[DataFolderStructure.COLOUR_CHECKER]:
            EVs = sorted(
                self.project_settings.data[DataFolderStructure.COLOUR_CHECKER].keys()
            )
            self._baseline_exposure = EVs[len(EVs) // 2]
            logger.warning(
                "Baseline exposure is different from zero: %s", self._baseline_exposure
            )

        paths = self.project_settings.data[DataFolderStructure.COLOUR_CHECKER][
            self._baseline_exposure
        ]
        with working_directory(self.project_settings.working_directory):
            logger.info(
                'Reading EV "%s" baseline exposure "ColourChecker" from "%s"...',
                self._baseline_exposure,
                paths[0],
            )
            image = _reformat_image(read_image(str(paths[0])))

        (
            rectangles,
            clusters,
            swatches,
            segmented_image,
        ) = segmenter_default(
            image,
            additional_data=True,
            **common.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
        ).values

        quadrilateral = rectangles[0]

        self._image_colour_checker_segmentation = np.copy(image)
        cv2.drawContours(
            self._image_colour_checker_segmentation, swatches, -1, (1, 0, 1), 3
        )
        cv2.drawContours(
            self._image_colour_checker_segmentation, clusters, -1, (0, 1, 1), 3
        )

        data_detection_colour_checker_EV0 = sample_colour_checker(
            image, quadrilateral, rectangle, common.SAMPLES_COUNT_DEFAULT, **settings
        )

        # Disabling orientation as we now have an oriented quadrilateral
        settings.reference_values = None

        # TODO Again are we using this?
        # Flatfield
        if self.project_settings.data.get("flatfield"):
            self._samples_analysis["flatfield"] = {"samples_sequence": []}
            for path in self.project_settings.data.get("flatfield", []):
                with working_directory(self.working_directory):
                    logger.info('Reading flatfield image from "%s"...', path)
                    image = _reformat_image(read_image(path))

                data_detection_flatfield = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    common.SAMPLES_COUNT_DEFAULT,
                    **settings,
                )

                self._samples_analysis["flatfield"]["samples_sequence"].append(
                    data_detection_flatfield.swatch_colours.tolist()
                )

            samples_sequence = as_float_array(
                [
                    samples[0]
                    for samples in self._samples_analysis["flatfield"][
                        "samples_sequence"
                    ]
                ]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis["flatfield"]["samples_median"] = np.median(
                as_float_array(self._samples_analysis["flatfield"]["samples_sequence"])[
                    mask
                ],
                (0, 1),
            ).tolist()

        # Grey Card
        if self.project_settings.data.get(DataFolderStructure.GREY_CARD, []):
            self._samples_analysis[DataFolderStructure.GREY_CARD] = {
                "samples_sequence": []
            }

            settings_grey_card = Structure(**settings)
            settings_grey_card.swatches_horizontal = 1
            settings_grey_card.swatches_vertical = 1

            for path in self.project_settings.data.get(
                DataFolderStructure.GREY_CARD, []
            ):
                with working_directory(self.project_settings.working_directory):
                    logger.info('Reading grey card image from "%s"...', path)
                    image = _reformat_image(read_image(path))

                data_detection_grey_card = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    common.SAMPLES_COUNT_DEFAULT,
                    **settings_grey_card,
                )

                grey_card_colour = np.ravel(data_detection_grey_card.swatch_colours)

                self._samples_analysis[DataFolderStructure.GREY_CARD][
                    "samples_sequence"
                ].append(grey_card_colour.tolist())

            samples_sequence = as_float_array(
                [
                    samples[0]
                    for samples in self._samples_analysis[
                        DataFolderStructure.GREY_CARD
                    ]["samples_sequence"]
                ]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis[DataFolderStructure.GREY_CARD][
                "samples_median"
            ] = np.median(
                as_float_array(
                    self._samples_analysis[DataFolderStructure.GREY_CARD][
                        "samples_sequence"
                    ]
                )[mask],
                (0, 1),
            ).tolist()

            self._image_grey_card_sampling = np.copy(image)
            image_grey_card_contour = zeros(
                (working_height, working_width), dtype=np.uint8
            )
            image_grey_card_contour[
                data_detection_grey_card.swatch_masks[0][
                    0
                ] : data_detection_grey_card.swatch_masks[0][1],
                data_detection_grey_card.swatch_masks[0][
                    2
                ] : data_detection_grey_card.swatch_masks[0][3],
                ...,
            ] = 255
            contours, _hierarchy = cv2.findContours(
                image_grey_card_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                self._image_grey_card_sampling,
                contours,
                -1,
                (1, 0, 1),
                3,
            )

        # ColourChecker Classic Samples per EV
        self._samples_analysis[DataFolderStructure.COLOUR_CHECKER] = {}
        for EV in self.project_settings.data[DataFolderStructure.COLOUR_CHECKER]:
            self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV] = {}
            self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                "samples_sequence"
            ] = []
            for path in self.project_settings.data[DataFolderStructure.COLOUR_CHECKER][
                EV
            ]:
                with working_directory(self.project_settings.working_directory):
                    logger.info(
                        'Reading EV "%s" "ColourChecker" from "%s"...',
                        EV,
                        path,
                    )

                    image = _reformat_image(read_image(path))

                data_detection_colour_checker = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    common.SAMPLES_COUNT_DEFAULT,
                    **settings,
                )

                self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                    "samples_sequence"
                ].append(data_detection_colour_checker.swatch_colours.tolist())

            sequence_neutral_5 = as_float_array(
                [
                    samples[21]
                    for samples in self._samples_analysis[
                        DataFolderStructure.COLOUR_CHECKER
                    ][EV]["samples_sequence"]
                ]
            )
            mask = np.all(~mask_outliers(sequence_neutral_5), axis=-1)

            self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                "samples_median"
            ] = np.median(
                as_float_array(
                    self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                        "samples_sequence"
                    ]
                )[mask],
                0,
            ).tolist()

        if self.project_settings.cleanup:
            shutil.rmtree(self.project_settings.working_directory)

    def sort(self):
        """
        Sort the samples produced by the image sampling process.

        The *ACES* reference samples are sorted and indexed as a function of the
        camera samples ordering. This ensures that the camera samples are
        monotonically increasing.

        Parameters
        ----------
        reference_colour_checker : NDArray
            Reference *ACES* *RGB* values for the *ColorChecker Classic*.
        """

        logger.info("Sorting camera and reference samples...")
        reference_colour_checker = self.project_settings.reference_colour_checker
        samples_camera = []
        samples_reference = []
        for EV, images in self._samples_analysis[
            DataFolderStructure.COLOUR_CHECKER
        ].items():
            samples_reference.append(reference_colour_checker[-6:, ...] * pow(2, EV))
            samples_EV = as_float_array(images["samples_median"])[-6:, ...]
            samples_camera.append(samples_EV)

        self._samples_camera = np.vstack(samples_camera)
        self._samples_reference = np.vstack(samples_reference)

        indices = np.argsort(np.median(self._samples_camera, axis=-1), axis=0)

        self._samples_camera = self._samples_camera[indices]
        self._samples_reference = self._samples_reference[indices]

    @abstractmethod
    def generate_LUT(self):
        """Implement the generation of the list and must populate the unfiltered lut"""

    @abstractmethod
    def filter_LUT(self):
        """Implement the filtering of the lut and must populate the filtered lut"""

    @abstractmethod
    def decode(self):
        """Implement the decoding of the lut"""

    @abstractmethod
    def optimise(self):
        """Implement any optimisation of the lut that is required"""

    def zip(
        self, output_directory, information=None, archive_serialised_generator=False
    ):
        """
        Zip the *Common LUT Format* (CLF) resulting from the *IDT* generation
        process.

        Parameters
        ----------
        output_directory : str
            Output directory for the *zip* file.
        information : dict
            Information pertaining to the *IDT* and the computation parameters.
        archive_serialised_generator : bool
            Whether to serialise and archive the *IDT* generator.

        Returns
        -------
        :class:`str`
            *Zip* file path.
        """
        # TODO There is a whole bunch of computation which happens within the ui to
        #  calculate things like the delta_e. All of that logic should be moved to the
        #  application or generator so we do not need to go out to the UI, to do
        #  calculations which then come back into application / generator in order
        #  for us to write it out
        information = information or {}

        logger.info(
            'Zipping the "CLF" resulting from the "IDT" generation '
            'process in "%s" output directory using given information: "%s".',
            output_directory,
            information,
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        camera_make = self.project_settings.project_settings.camera_make
        camera_model = self.project_settings.project_settings.camera_model

        clf_path = self.to_clf(output_directory, information)

        json_path = f"{output_directory}/{camera_make}.{camera_model}.json"
        with open(json_path, "w") as json_file:
            json_file.write(jsonpickle.encode(self, indent=2))

        zip_file = Path(output_directory) / f"IDT_{camera_make}_{camera_model}.zip"
        current_working_directory = os.getcwd()
        try:
            os.chdir(output_directory)
            with ZipFile(zip_file, "w") as zip_archive:
                zip_archive.write(clf_path.replace(output_directory, "")[1:])
                if archive_serialised_generator:
                    zip_archive.write(json_path.replace(output_directory, "")[1:])
        finally:
            os.chdir(current_working_directory)

        return zip_file

    def to_clf(self, output_directory, information):
        """
        Convert the *IDT* generation process data to *Common LUT Format* (CLF).

        Parameters
        ----------
        output_directory : str
            Output directory for the zip file.
        information : dict
            Information pertaining to the *IDT* and the computation parameters.

        Returns
        -------
        :class:`str`
            *CLF* file path.
        """

        logger.info(
            'Converting "IDT" generation process data to "CLF" in "%s"'
            'output directory using given information: "%s".',
            output_directory,
            information,
        )

        project_settings = self.project_settings.project_settings

        aces_transform_id = project_settings.aces_transform_id
        aces_user_name = project_settings.aces_user_name
        camera_make = project_settings.camera_make
        camera_model = project_settings.camera_model

        root = Et.Element(
            "ProcessList",
            compCLFversion="3",
            id=aces_transform_id,
            name=aces_user_name,
        )

        def format_array(a):
            """Format given array :math:`a`."""

            return re.sub(r"\[|\]|,", "", "\n".join(map(str, a.tolist())))

        et_input_descriptor = Et.SubElement(root, "InputDescriptor")
        et_input_descriptor.text = f"{camera_make} {camera_model}"

        et_output_descriptor = Et.SubElement(root, "OutputDescriptor")
        et_output_descriptor.text = "ACES2065-1"

        et_info = Et.SubElement(root, "Info")
        et_metadata = Et.SubElement(et_info, "Archive")
        for key, prop in project_settings.properties:
            value = prop.getter(project_settings)
            if key == "schema_version":
                continue

            sub_element = Et.SubElement(
                et_metadata, key.replace("_", " ").title().replace(" ", "")
            )
            sub_element.text = str(value)
        et_academy_idt_calculator = Et.SubElement(et_info, "AcademyIDTCalculator")
        for key, value in information.items():
            sub_element = Et.SubElement(et_academy_idt_calculator, key)
            sub_element.text = str(value)

        et_lut = Et.SubElement(
            root,
            "LUT1D",
            inBitDepth="32f",
            outBitDepth="32f",
            interpolation="linear",
        )
        LUT_decoding = self._LUT_decoding
        channels = 1 if isinstance(LUT_decoding, LUT1D) else 3
        et_description = Et.SubElement(et_lut, "Description")
        et_description.text = f"Linearisation *{LUT_decoding.__class__.__name__}*."
        et_array = Et.SubElement(et_lut, "Array", dim=f"{LUT_decoding.size} {channels}")
        et_array.text = f"\n{format_array(LUT_decoding.table)}"

        root = clf_processing_elements(root, self._M, self._RGB_w, self._k, False)

        clf_path = (
            f"{output_directory}/"
            f"{camera_make}.Input.{camera_model}_to_ACES2065-1.clf"
        )
        Et.indent(root)

        with open(clf_path, "w") as clf_file:
            clf_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            clf_file.write(Et.tostring(root, encoding="UTF-8").decode("utf8"))

        return clf_path

    def png_colour_checker_segmentation(self):
        """
        Return the colour checker segmentation image as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """

        if self._image_colour_checker_segmentation is None:
            return None

        colour.plotting.plot_image(
            self._image_colour_checker_segmentation,
            show=False,
        )

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

    def png_grey_card_sampling(self):
        """
        Return the grey card image sampling as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """

        if self._image_grey_card_sampling is None:
            return None

        colour.plotting.plot_image(
            self._image_grey_card_sampling,
            show=False,
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

    def png_extrapolated_camera_samples(self):
        """
        Return the extrapolated camera samples as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """
        raise NotImplementedError(
            "png_extrapolated_camera_samples method not implemented"
        )
