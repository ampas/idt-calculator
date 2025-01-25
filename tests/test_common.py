"""Define the unit tests for the :mod:`aces.idt.core.common` module."""

from __future__ import annotations

import json
import os.path

import numpy as np

from aces.idt.core import EXPOSURE_CLIPPING_THRESHOLD
from aces.idt.core.common import (
    calculate_camera_npm_and_primaries_wp,
    find_clipped_exposures,
    find_similar_rows,
    generate_reference_colour_checker,
)
from tests.test_utils import TestIDTBase

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "TestGenerateReferenceColourChecker",
    "TestCalculateCameraNpmAndPrimariesWp",
    "TestFindSimilarRows",
    "TestFindClippedExposures",
]


class TestGenerateReferenceColourChecker:
    """
    Define :func:`aces.idt.core.common.generate_reference_colour_checker`
    definition unit tests methods.
    """

    def test_generate_reference_colour_checker(self) -> None:
        """
        Test :func:`aces.idt.core.common.generate_reference_colour_checker`
        definition.
        """

        np.testing.assert_allclose(
            generate_reference_colour_checker(),
            [
                [0.11876842177626151, 0.08709058563389999, 0.058951218439182544],
                [0.40000355538786564, 0.31916517742981565, 0.23733683167215208],
                [0.18476317529262326, 0.20397604937400404, 0.3130841414707486],
                [0.10900767177832925, 0.13511034887945872, 0.06492296526026398],
                [0.2668325426084729, 0.2460383081243736, 0.40929373504647487],
                [0.3228275347098078, 0.46209038705471595, 0.4059997268840585],
                [0.3860420994813448, 0.22743356338910722, 0.05776644858328035],
                [0.1382168532394208, 0.1303655057792208, 0.3369849589807394],
                [0.30200029194805983, 0.13751934107949534, 0.12757993019390476],
                [0.09309755810948224, 0.06346469265430443, 0.13526731719711416],
                [0.34876526154496384, 0.43654731605432173, 0.10611841964202463],
                [0.4865437930159921, 0.3668514610866347, 0.08060352133648069],
                [0.08730558671548493, 0.07442476376021454, 0.2726781684209821],
                [0.15366178743784017, 0.25691653578416973, 0.09070349092374685],
                [0.21739211063561842, 0.07069630521117498, 0.05129818513944815],
                [0.5891828043214191, 0.5394372192137751, 0.09155549948424543],
                [0.3090146231057637, 0.14817408797862763, 0.2742627591798639],
                [0.14899905555111928, 0.23378396576462934, 0.3593217013366608],
                [0.8665068310764097, 0.8679173464028738, 0.858093881322456],
                [0.5735498956190463, 0.5725588746417859, 0.5716389345559676],
                [0.3534557219490971, 0.3533663496431708, 0.353882100057212],
                [0.20252229397603771, 0.2024315233275667, 0.20285713026326946],
                [0.0946720142743027, 0.0952034139165298, 0.0963649584715441],
                [0.03745083433881645, 0.03766169506735516, 0.038954301107662154],
            ],
            atol=1e-15,
        )


class TestCalculateCameraNpmAndPrimariesWp:
    """
    Define :func:`aces.idt.core.common.calculate_camera_npm_and_primaries_wp`
    definition unit tests methods.
    """

    def test_calculate_camera_npm_and_primaries_wp(self) -> None:
        """
        Test :func:`aces.idt.core.common.calculate_camera_npm_and_primaries_wp`
        definition.
        """

        input_matrix = [
            [0.785043, 0.083844, 0.131118],
            [0.023172, 1.087892, -0.111055],
            [-0.073769, -0.314639, 1.388537],
        ]

        npm, primaries, wp = calculate_camera_npm_and_primaries_wp(input_matrix)

        expected_npm = [
            [0.7353579, 0.06867992, 0.14646275],
            [0.28673187, 0.84296573, -0.1297009],
            [-0.07965591, -0.34720223, 1.5155319],
        ]

        expected_primaries = [
            [0.7802753326723714, 0.3042461485259778],
            [0.1216772523298739, 1.4934459215643787],
            [0.09558399073520502, -0.08464493023427101],
        ]

        expected_wp = [0.31274994, 0.32903601]

        np.testing.assert_allclose(expected_npm, npm, atol=1e-6)
        np.testing.assert_allclose(expected_primaries, primaries, atol=1e-6)
        np.testing.assert_allclose(expected_wp, wp, atol=1e-6)


class TestFindSimilarRows(TestIDTBase):
    """
    Define :func:`aces.idt.core.common.find_similar_rows` definition unit tests
    methods.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        self.tolerance = 0.005

    def test_scenario_1(self) -> None:
        """
        Test case where none of the values are lower than the threshold and
        ensure that no values are masked.
        """

        colour_checker_scenario_1 = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
                [[0.25, 0.35, 0.45], [0.55, 0.65, 0.75], [0.85, 0.95, 1.05]],
                [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
            ]
        )
        clipped_indices = find_similar_rows(
            colour_checker_scenario_1, threshold=self.tolerance
        )
        expected_data = []
        np.testing.assert_array_equal(clipped_indices, expected_data)

    def test_scenario_2(self) -> None:
        """
        Test case where the first two values are clipped and ensure that the
        first row is masked.
        """

        colour_checker_scenario_2 = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.5, 0.5, 0.5],
                [0.6, 0.6, 0.6],
                [0.7, 0.7, 0.7],
            ]
        )
        clipped_indices = find_similar_rows(
            colour_checker_scenario_2, threshold=self.tolerance
        )
        expected_data = [0]
        self.assertEqual(clipped_indices, expected_data)

    def test_scenario_3(self) -> None:
        """
        Test case where the last two values are clipped and ensure that the
        last row is masked.
        """

        colour_checker_scenario_3 = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.15, 0.25, 0.35],
                [0.2, 0.3, 0.4],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
            ]
        )
        clipped_indices = find_similar_rows(
            colour_checker_scenario_3, threshold=self.tolerance
        )
        expected_data = [4]
        self.assertEqual(clipped_indices, expected_data)

    def test_scenario_4(self) -> None:
        """
        Test case where both the top and bottom values are clipped and ensure
        that both the first and last rows are masked.
        """

        colour_checker_scenario_4 = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.2, 0.3, 0.4],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        clipped_indices = find_similar_rows(
            colour_checker_scenario_4, threshold=self.tolerance
        )
        expected_data = [0, 4]
        np.testing.assert_array_equal(clipped_indices, expected_data)

    def test_scenario_5(self) -> None:
        """
        Test case where both the top and bottom values are clipped with multiple
        instances and ensure that the first two and last rows are masked.
        """

        colour_checker_scenario_5 = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.15, 0.15, 0.15],
                [0.2, 0.3, 0.4],
                [0.9, 0.9, 0.9],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        clipped_indices = find_similar_rows(
            colour_checker_scenario_5, threshold=self.tolerance
        )
        expected_data = [0, 1, 7]
        np.testing.assert_array_equal(clipped_indices, expected_data)


class TestFindClippedExposures(TestIDTBase):
    """
    Define :func:`aces.idt.core.common.find_clipped_exposures` definition unit
    tests methods.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        self.tolerance = 0.0499

    def test_scenario_1(self) -> None:
        """
        Test the scenario where none of the exposures haves values below the
        threshold and, thus, ensure that no exposures values were removed.
        """

        colour_checker_scenario_1 = {
            -2.0: np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            -1.0: np.array(
                [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]]
            ),
            0.0: np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]),
            1.0: np.array([[0.25, 0.35, 0.45], [0.55, 0.65, 0.75], [0.85, 0.95, 1.05]]),
            2.0: np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]),
        }

        removed_evs = find_clipped_exposures(colour_checker_scenario_1, self.tolerance)
        expected_ev_keys = []
        self.assertEqual(removed_evs, expected_ev_keys)

    def test_scenario_2(self) -> None:
        """
        Test the scenario where exposures -2.0 and -1.0 have values below the
        threshold and, thus, exposure -2 should be removed.
        """

        colour_checker_scenario_2 = {
            -2.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            -1.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            0.0: np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            1.0: np.array([[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]]),
            2.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
        }

        removed_evs = find_clipped_exposures(colour_checker_scenario_2, self.tolerance)
        expected_ev_keys = [-2.0]
        self.assertEqual(sorted(removed_evs), expected_ev_keys)

    def test_scenario_3(self) -> None:
        """
        Test the scenario where exposures 1.0 and 2.0 have values below the
        threshold and, thus, exposure 2 should be removed.
        """

        colour_checker_scenario_3 = {
            -2.0: np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            -1.0: np.array(
                [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]]
            ),
            0.0: np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]),
            1.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            2.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
        }

        removed_evs = find_clipped_exposures(colour_checker_scenario_3, self.tolerance)
        expected_ev_keys = [2.0]
        self.assertEqual(removed_evs, expected_ev_keys)

    def test_scenario_4(self) -> None:
        """
        Test the scenario where both the bottom and top exposures have values
        below the threshold and, thus, exposures -2 and 2 should be removed.
        """

        colour_checker_scenario_4 = {
            -2.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            -1.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            0.0: np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
            1.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
            2.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
        }

        removed_evs = find_clipped_exposures(colour_checker_scenario_4, self.tolerance)
        expected_ev_keys = [-2.0, 2.0]
        self.assertEqual(removed_evs, expected_ev_keys)

    def test_scenario_5(self) -> None:
        """
        Test the scenario where exposures -3, -2, 2.0, and, 3 should be removed.
        """

        colour_checker_scenario_5 = {
            -3.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            -2.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            -1.0: np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            0.0: np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 2.0]]),
            1.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
            2.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
            3.0: np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]]),
        }

        removed_evs = find_clipped_exposures(colour_checker_scenario_5, self.tolerance)
        expected_ev_keys = [-3.0, -2.0, 2.0, 3.0]
        self.assertEqual(removed_evs, expected_ev_keys)

    def test_scenario_6(self) -> None:
        """
        Test the scenario where exposures -6, -5, -4, 4, 5, and, 6 should be
        removed.
        """

        file_path = os.path.join(
            self.get_test_resources_folder(), "samples_decoded_clipped.json"
        )

        with open(file_path) as file:
            temp_dict = json.load(file)
            colour_checker_scenario_6 = {}
            for key, value in temp_dict.items():
                colour_checker_scenario_6[float(key)] = np.array(value)
        removed_evs = find_clipped_exposures(
            colour_checker_scenario_6, EXPOSURE_CLIPPING_THRESHOLD
        )
        expected_ev_keys = [-6.0, -5.0, -4.0, 4.0, 5.0, 6.0]
        self.assertEqual(removed_evs, expected_ev_keys)
