"""
Module holds a common base generator class for which all other IDT generators inherit from
"""
import logging
import shutil
from copy import deepcopy

import cv2
import numpy as np
from colour import read_image
from colour.utilities import Structure, as_float_array
from colour_checker_detection import segmenter_default
from colour_checker_detection.detection import reformat_image, as_int32_array, sample_colour_checker
from numpy import zeros

from idt.core import common
from idt.core.constants import DataFolderStructure
from idt.core.utilities import working_directory, mask_outliers

logger = logging.getLogger(__name__)


class BaseGenerator:
    def __init__(self, application):
        self._application = application
        self._projectSettings = application.project_settings
        self._samples_analysis = None

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

        self._samples_analysis = deepcopy(self._projectSettings.data)

        # Baseline exposure value, it can be different from zero.
        if 0 not in self._projectSettings.data[DataFolderStructure.COLOUR_CHECKER]:
            EVs = sorted(self._projectSettings.data[DataFolderStructure.COLOUR_CHECKER].keys())
            self._baseline_exposure = EVs[len(EVs) // 2]
            logger.warning(
                "Baseline exposure is different from zero: %s", self._baseline_exposure
            )

        paths = self._projectSettings.data[DataFolderStructure.COLOUR_CHECKER][self._baseline_exposure]
        with working_directory(self._application.working_directory):
            logger.info(
                'Reading EV "%s" baseline exposure "ColourChecker" from "%s"...',
                self._baseline_exposure,
                paths[0],
            )
            image = _reformat_image(read_image(paths[0]))

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
        if self._projectSettings.data.get("flatfield"):
            self._samples_analysis["flatfield"] = {"samples_sequence": []}
            for path in self._projectSettings.data.get("flatfield", []):
                with working_directory(self._application.working_directory):
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
                [samples[0] for samples in self._samples_analysis["flatfield"]["samples_sequence"]]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis["flatfield"]["samples_median"] = np.median(
                as_float_array(
                    self._samples_analysis["flatfield"]["samples_sequence"]
                )[mask],
                (0, 1),
            ).tolist()

        # Grey Card
        if self._projectSettings.data.get(DataFolderStructure.GREY_CARD, []):
            self._samples_analysis[DataFolderStructure.GREY_CARD] = {"samples_sequence": []}

            settings_grey_card = Structure(**settings)
            settings_grey_card.swatches_horizontal = 1
            settings_grey_card.swatches_vertical = 1

            for path in self._projectSettings.data.get(DataFolderStructure.GREY_CARD, []):
                with working_directory(self._application.working_directory):
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

                self._samples_analysis[DataFolderStructure.GREY_CARD]["samples_sequence"].append(
                    grey_card_colour.tolist()
                )

            samples_sequence = as_float_array(
                [samples[0] for samples in self._samples_analysis[DataFolderStructure.GREY_CARD]["samples_sequence"]]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis[DataFolderStructure.GREY_CARD]["samples_median"] = np.median(
                as_float_array(
                    self._samples_analysis[DataFolderStructure.GREY_CARD]["samples_sequence"]
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
            ]: data_detection_grey_card.swatch_masks[0][1],
            data_detection_grey_card.swatch_masks[0][
                2
            ]: data_detection_grey_card.swatch_masks[0][3],
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
        for EV in self._projectSettings.data[DataFolderStructure.COLOUR_CHECKER]:
            self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV] = {}
            self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                "samples_sequence"
            ] = []
            for path in self._projectSettings.data[DataFolderStructure.COLOUR_CHECKER][EV]:
                with working_directory(self._application.working_directory):
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
                    for samples in self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                    "samples_sequence"
                ]
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

        if self._application.cleanup:
            shutil.rmtree(self._application.working_directory)

    def generate_LUT(self):
        raise NotImplementedError("generate_LUT method not implemented")

    def filter_LUT(self):
        raise NotImplementedError("filter_LUT method not implemented")

    def decode(self):
        raise NotImplementedError("decode method not implemented")

    def optimise(self):
        raise NotImplementedError("optimise method not implemented")
