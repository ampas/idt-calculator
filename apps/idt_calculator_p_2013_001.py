"""
Input Device Transform (IDT) Calculator - P-2013-001
====================================================
"""

import colour
import numpy as np
import sys
import urllib.parse
from colour import (
    RGB_COLOURSPACES,
    RGB_to_RGB,
    SDS_ILLUMINANTS,
    XYZ_to_RGB,
    SpectralDistribution,
    sd_CIE_illuminant_D_series,
    sd_blackbody,
)
from colour.characterisation import (
    RGB_CameraSensitivities,
    camera_RGB_to_ACES2065_1,
    matrix_idt,
)
from colour.models import RGB_COLOURSPACE_ACES2065_1
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (
    CACHE_REGISTRY,
    as_float,
    as_float_array,
    numpy_print_options,
)
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from dash.dcc import Link, Location, Markdown
from dash.dependencies import Input, Output, State
from dash.html import A, Code, Div, Footer, H3, Img, Li, Main, Pre, Ul
from dash_bootstrap_components import (
    Button,
    Card,
    CardBody,
    CardHeader,
    Col,
    Collapse,
    Container,
    InputGroup,
    InputGroupText,
    Row,
    Select,
    Tab,
    Tabs,
    Tooltip,
)

# "Input" is already imported above, to avoid clash, we alias it as "Field".
from dash_bootstrap_components import Input as Field
from datetime import datetime

from aces.idt import png_compare_colour_checkers, error_delta_E, slugify
from app import APP, SERVER_URL, __version__
from apps.common import (
    OPTIONS_CAMERA_SENSITIVITIES,
    OPTIONS_CAT,
    COLOUR_ENVIRONMENT,
    CUSTOM_WAVELENGTHS,
    DATATABLE_DECIMALS,
    OPTIMISATION_FACTORIES,
    OPTIONS_DISPLAY_COLOURSPACES,
    OPTIONS_ILLUMINANT,
    OPTIONS_INTERPOLATION,
    OPTIONS_OPTIMISATION_SPACES,
    INTERPOLATORS,
    MSDS_CAMERA_SENSITIVITIES,
    STYLE_DATATABLE,
    TEMPLATE_DEFAULT_OUTPUT,
    TEMPLATE_CTL_MODULE,
    TEMPLATE_DCTL_MODULE,
    TEMPLATE_NUKE_GROUP,
    TRAINING_DATA_KODAK190PATCHES,
    format_float,
    format_idt_clf,
    format_matrix_ctl,
    format_vector_nuke,
    format_vector_ctl,
    format_matrix_nuke,
    format_matrix_dctl,
    format_vector_dctl,
)

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "APP_NAME",
    "APP_PATH",
    "APP_DESCRIPTION",
    "APP_UID",
    "LAYOUT",
    "set_camera_sensitivities_datable",
    "set_illuminant_datable",
    "toggle_advanced_options",
    "compute_idt_p2013_001",
]

APP_NAME = "Academy Input Device Transform (IDT) Calculator - P-2013-001"
"""
App name.

APP_NAME : str
"""

APP_PATH = f"/apps/{__name__.split('.')[-1]}"
"""
App path, i.e. app url.

APP_PATH : str
"""

APP_DESCRIPTION = (
    "This app computes the *Input Device Transform* (IDT) "
    "for given camera sensitivities and illuminant according to *P-2013-001* "
    "method."
)
"""
App description.

APP_DESCRIPTION : str
"""

APP_UID = hash(APP_NAME)
"""
App unique id.

APP_UID : str
"""

_OPTIONS_TRAINING_DATASET = [
    {"label": key, "value": key}
    for key in ["Kodak - 190 Patches", "ISO 17321-1"]
]

_TRAINING_DATASETS = {
    "Kodak - 190 Patches": TRAINING_DATA_KODAK190PATCHES,
    "ISO 17321-1": colour.colorimetry.sds_and_msds_to_msds(
        colour.SDS_COLOURCHECKERS["ISO 17321-1"].values()
    ),
}

_TRAINING_DATASET_TO_COLUMNS = {
    "Kodak - 190 Patches": 19,
    "ISO 17321-1": 6,
}


_OPTIONS_FORMATTER = [
    {"label": label, "value": value}
    for label, value in [
        ("Str", "str"),
        ("Repr", "repr"),
        ("CLF", "clf"),
        ("CTL", "ctl"),
        ("DCTL", "dctl"),
        ("Nuke", "nuke"),
    ]
]

_CACHE_MATRIX_IDT = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_MATRIX_IDT"
)


def _uid(id_):
    """
    Generate a unique id for given id by appending the application *UID*.
    """

    return f"{id_}-{APP_UID}"


_LAYOUT_COLUMN_CAMERA_SENSITIVITIES_CHILDREN = [
    InputGroup(
        [
            InputGroupText("Camera Sensitivities"),
            Select(
                id=_uid("camera-sensitivities-select"),
                options=OPTIONS_CAMERA_SENSITIVITIES,
                value=OPTIONS_CAMERA_SENSITIVITIES[0]["value"],
            ),
        ],
        className="mb-1",
    ),
    Tooltip(
        "Camera sensitivities used to integrate the incident reflectance "
        'training datasets. External tabular data, e.g. from "Excel" or '
        '"Google Docs" can be pasted directly.',
        delay={"show": 500, "hide": 500},
        target=_uid("camera-sensitivities-select"),
    ),
    Row(
        Col(
            DataTable(
                id=_uid("camera-sensitivities-datatable"),
                editable=True,
                style_as_list_view=True,
                style_header={
                    "backgroundColor": STYLE_DATATABLE[
                        "header_background_colour"
                    ]
                },
                style_cell={
                    "backgroundColor": STYLE_DATATABLE[
                        "cell_background_colour"
                    ],
                    "color": STYLE_DATATABLE["cell_colour"],
                },
            ),
        ),
    ),
]

_LAYOUT_COLUMN_ILLUMINANT_CHILDREN = [
    InputGroup(
        [
            InputGroupText("Illuminant"),
            Select(
                id=_uid("illuminant-select"),
                options=OPTIONS_ILLUMINANT,
                value=OPTIONS_ILLUMINANT[0]["value"],
            ),
        ],
        className="mb-1",
    ),
    Tooltip(
        "Illuminant used to produce the incident reflectance training and "
        'test datasets. Selecting "Daylight" and "Blackbody" displays a new '
        "input field allowing to define a custom colour temperature. It is "
        "possible to paste external tabular data.",
        delay={"show": 500, "hide": 500},
        target=_uid("illuminant-select"),
    ),
    Row(
        Col(
            [
                Collapse(
                    InputGroup(
                        [
                            InputGroupText("CCT"),
                            Field(
                                id=_uid("cct-field"),
                                type="number",
                                value=5500,
                            ),
                        ],
                        className="mb-1",
                    ),
                    id=_uid("illuminant-options-collapse"),
                    className="mb-1",
                ),
                DataTable(
                    id=_uid("illuminant-datatable"),
                    editable=True,
                    style_as_list_view=True,
                    style_header={
                        "backgroundColor": STYLE_DATATABLE[
                            "header_background_colour"
                        ]
                    },
                    style_cell={
                        "backgroundColor": STYLE_DATATABLE[
                            "cell_background_colour"
                        ],
                        "color": STYLE_DATATABLE["cell_colour"],
                    },
                ),
            ]
        ),
    ),
]

_LAYOUT_COLUMN_OPTIONS_CHILDREN = [
    Card(
        [
            CardHeader("Options"),
            CardBody(
                [
                    Button(
                        "Toggle Advanced Options",
                        id=_uid("toggle-advanced-options-button"),
                        className="mb-2",
                        style={"width": "100%"},
                    ),
                    Collapse(
                        [
                            InputGroup(
                                [
                                    InputGroupText("RGB Display Colourspace"),
                                    Select(
                                        id=_uid(
                                            "rgb-display-colourspace-select"
                                        ),
                                        options=OPTIONS_DISPLAY_COLOURSPACES,
                                        value=OPTIONS_DISPLAY_COLOURSPACES[0][
                                            "value"
                                        ],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "RGB colourspace used to display images "
                                "in the app. It does not affect the "
                                "computations.",
                                delay={"show": 500, "hide": 500},
                                target=_uid("rgb-display-colourspace-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Training Data"),
                                    Select(
                                        id=_uid("training-data-select"),
                                        options=_OPTIONS_TRAINING_DATASET,
                                        value=(
                                            _OPTIONS_TRAINING_DATASET[0][
                                                "value"
                                            ]
                                        ),
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Reflectance training dataset used for the "
                                "computations. A larger and varied training "
                                'dataset produces a better "IDT".',
                                delay={"show": 500, "hide": 500},
                                target=_uid("training-data-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("CAT"),
                                    Select(
                                        id=_uid(
                                            "chromatic-adaptation-transform-select"
                                        ),
                                        options=OPTIONS_CAT,
                                        value=str(OPTIONS_CAT[3]["value"]),
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Chromatic adaptation transform used to convert "
                                "the reflectance training dataset under the "
                                '"ACES" whitepoint.',
                                delay={"show": 500, "hide": 500},
                                target=_uid(
                                    "chromatic-adaptation-transform-select"
                                ),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Optimisation Space"),
                                    Select(
                                        id=_uid("optimisation-space-select"),
                                        options=OPTIONS_OPTIMISATION_SPACES,
                                        value=OPTIONS_OPTIMISATION_SPACES[0][
                                            "value"
                                        ],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Colour model used to compute the error "
                                "during the optimisation process. Recent "
                                'models such as "Oklab" and "JzAzBz" tend to '
                                "produce a lower error.",
                                delay={"show": 500, "hide": 500},
                                target=_uid("optimisation-space-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText(
                                        "Camera Sensitivities Interpolator"
                                    ),
                                    Select(
                                        id=_uid(
                                            "camera-sensitivities-interpolator-select"
                                        ),
                                        options=OPTIONS_INTERPOLATION,
                                        value=OPTIONS_INTERPOLATION[3][
                                            "value"
                                        ],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Interpolator used to align the camera "
                                "sensitivities to the working spectral shape, "
                                "i.e. `colour.SpectralShape(380, 780, 5)`",
                                delay={"show": 500, "hide": 500},
                                target=_uid(
                                    "camera-sensitivities-interpolator-select"
                                ),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Illuminant Interpolator"),
                                    Select(
                                        id=_uid(
                                            "illuminant-interpolator-select"
                                        ),
                                        options=OPTIONS_INTERPOLATION,
                                        value=OPTIONS_INTERPOLATION[1][
                                            "value"
                                        ],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Interpolator used to align the illuminant "
                                "to the working spectral shape, i.e. "
                                "`colour.SpectralShape(380, 780, 5)`",
                                delay={"show": 500, "hide": 500},
                                target=_uid("illuminant-interpolator-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Exposure Factor"),
                                    Field(
                                        id=_uid("exposure-factor-select"),
                                        type="number",
                                        value=1,
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                'Exposure factor "k" that results in a '
                                'nominally "18% gray" object in the scene '
                                "producing ACES values [0.18, 0.18, 0.18].",
                                delay={"show": 500, "hide": 500},
                                target=_uid("exposure-factor-select"),
                            ),
                        ],
                        id=_uid("advanced-options-collapse"),
                        className="mb-1",
                    ),
                    InputGroup(
                        [
                            InputGroupText("Formatter"),
                            Select(
                                id=_uid("formatter-select"),
                                options=_OPTIONS_FORMATTER,
                                value="str",
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        'Formatter used to generate the "IDT".',
                        delay={"show": 500, "hide": 500},
                        target=_uid("formatter-select"),
                    ),
                    InputGroup(
                        [
                            InputGroupText("Decimals"),
                            Select(
                                id=_uid("decimals-select"),
                                options=[
                                    {"label": str(a), "value": a}
                                    for a in range(1, 16, 1)
                                ],
                                value=10,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        'Decimals used in the formatted "IDT".',
                        delay={"show": 500, "hide": 500},
                        target=_uid("decimals-select"),
                    ),
                ]
            ),
        ],
        className="mb-2",
    ),
    Card(
        [
            CardHeader("Input Device Transform"),
            CardBody(
                [
                    Row(
                        [
                            Col(
                                Button(
                                    "Compute IDT",
                                    id=_uid("compute-idt-button"),
                                    style={"width": "100%"},
                                )
                            ),
                            Col(
                                Button(
                                    "Copy to Clipboard",
                                    id=_uid("copy-to-clipboard-button"),
                                    style={"width": "100%"},
                                )
                            ),
                        ]
                    ),
                    Pre(
                        Code(
                            id=_uid("idt-calculator-output"),
                            className="code shell",
                        ),
                        id=_uid("idt-calculator-pre"),
                        className="mt-2",
                        style={"display": "none"},
                    ),
                ]
            ),
        ]
    ),
    Row(
        Col(
            Div(id=_uid("output-data-div")),
        ),
    ),
]

_LAYOUT_COLUMN_FOOTER_CHILDREN = [
    Ul(
        [
            Li(
                Link("Back to index...", href="/", className="app-link"),
                className="list-inline-item",
            ),
            Li(
                A(
                    "Permalink",
                    href=urllib.parse.urljoin(SERVER_URL, APP_PATH),
                    target="_blank",
                ),
                className="list-inline-item",
            ),
            Li(
                A(
                    "ACES Central",
                    href="https://acescentral.com/",
                    target="_blank",
                ),
                className="list-inline-item",
            ),
        ],
        className="list-inline mt-3",
    ),
    Div(id=_uid("dev-null"), style={"display": "none"}),
]

LAYOUT = Container(
    [
        H3([Link(APP_NAME, href=APP_PATH)]),
        Location(id=_uid("url"), refresh=False),
        Main(
            Tabs(
                [
                    Tab(
                        Row(
                            [
                                Col(
                                    _LAYOUT_COLUMN_CAMERA_SENSITIVITIES_CHILDREN,
                                    width=4,
                                ),
                                Col(
                                    _LAYOUT_COLUMN_ILLUMINANT_CHILDREN,
                                    width=3,
                                ),
                                Col(
                                    _LAYOUT_COLUMN_OPTIONS_CHILDREN,
                                    width=5,
                                ),
                            ]
                        ),
                        label="Computations",
                        className="mt-3",
                    ),
                    Tab(
                        [
                            Markdown(APP_DESCRIPTION),
                            Markdown(f"{APP_NAME} - {__version__}"),
                            Pre(
                                [
                                    Code(
                                        COLOUR_ENVIRONMENT,
                                        className="code shell",
                                    )
                                ]
                            ),
                        ],
                        label="About",
                        className="mt-3",
                    ),
                ]
            ),
        ),
        Footer(
            Container(
                Row(
                    Col(_LAYOUT_COLUMN_FOOTER_CHILDREN),
                    className="text-center",
                ),
                fluid=True,
            ),
            className="footer",
        ),
    ]
)
"""
App layout, i.e. :class:`Container` class instance.

LAYOUT : Div
"""


@APP.callback(
    [
        Output(
            component_id=_uid("camera-sensitivities-datatable"),
            component_property="data",
        ),
        Output(
            component_id=_uid("camera-sensitivities-datatable"),
            component_property="columns",
        ),
    ],
    [Input(_uid("camera-sensitivities-select"), "value")],
)
def set_camera_sensitivities_datable(camera_sensitivities):
    """
    Set the *Camera Sensitivities* `DataTable` content for given camera
    sensitivities name.

    Parameters
    ----------
    camera_sensitivities : str
        Existing camera sensitivities name or *Custom*.

    Returns
    -------
    tuple
        Tuple of data and columns.
    """

    labels = ["Wavelength", "Red", "Green", "Blue"]
    ids = ["wavelength", "R", "G", "B"]
    precision = [None] + [
        Format(precision=DATATABLE_DECIMALS, scheme=Scheme.fixed)
    ] * 3
    columns = [
        {
            "id": ids[i],
            "name": label,
            "type": "numeric",
            "format": precision[i],
        }
        for i, label in enumerate(labels)
    ]

    if camera_sensitivities == "Custom":
        data = [
            dict(wavelength=wavelength, **{column: None for column in labels})
            for wavelength in CUSTOM_WAVELENGTHS
        ]

    else:
        camera_sensitivities = MSDS_CAMERA_SENSITIVITIES[camera_sensitivities]

        data = [
            dict(
                wavelength=wavelength,
                **{
                    column: camera_sensitivities.signals[column][wavelength]
                    for column in camera_sensitivities.labels
                },
            )
            for wavelength in camera_sensitivities.wavelengths
        ]

    return data, columns


@APP.callback(
    [
        Output(
            component_id=_uid("illuminant-datatable"),
            component_property="data",
        ),
        Output(
            component_id=_uid("illuminant-datatable"),
            component_property="columns",
        ),
    ],
    [
        Input(_uid("illuminant-select"), "value"),
        Input(_uid("cct-field"), "value"),
    ],
)
def set_illuminant_datable(illuminant, CCT):
    """
    Set the *Illuminant* `DataTable` content for given illuminant name.

    Parameters
    ----------
    illuminant : str
        Existing illuminant name or *Custom*, *Daylight* or *Blackbody*.
    CCT : numeric
        Custom correlated colour temperature (CCT) used for the *Daylight* and
        *Blackbody* illuminant types.

    Returns
    -------
    tuple
        Tuple of data and columns.
    """

    labels = ["Wavelength", "Irradiance"]
    ids = ["wavelength", "irradiance"]
    precision = [
        None,
        Format(precision=DATATABLE_DECIMALS, scheme=Scheme.fixed),
    ]
    columns = [
        {
            "id": ids[i],
            "name": label,
            "type": "numeric",
            "format": precision[i],
        }
        for i, label in enumerate(labels)
    ]

    if illuminant == "Custom":
        data = [
            dict(wavelength=wavelength, **{"irradiance": None})
            for wavelength in CUSTOM_WAVELENGTHS
        ]

    else:
        if illuminant == "Daylight":
            xy = CCT_to_xy_CIE_D(CCT * 1.4388 / 1.4380)
            illuminant = sd_CIE_illuminant_D_series(xy)
        elif illuminant == "Blackbody":
            illuminant = sd_blackbody(CCT)
        else:
            illuminant = SDS_ILLUMINANTS[illuminant]

        data = [
            dict(
                wavelength=wavelength,
                **{"irradiance": illuminant[wavelength]},
            )
            for wavelength in illuminant.wavelengths
        ]

    return data, columns


@APP.callback(
    Output(_uid("illuminant-options-collapse"), "is_open"),
    [Input(_uid("illuminant-select"), "value")],
    [State(_uid("illuminant-options-collapse"), "is_open")],
)
def toggle_options_illuminant(illuminant, is_open):
    """
    Collapse the *Illuminant Options* `Collapse` panel according to the
    selected illuminant type.

    Parameters
    ----------
    illuminant : str
        Existing illuminant name or *Custom*, *Daylight* or *Blackbody*.
    is_open : bool
        Whether the *Advanced Options* `Collapse` panel is opened or collapsed.

    Returns
    -------
    bool
        Whether to open or collapse the *Illuminant Options* `Collapse` panel.
    """

    return illuminant in ("Daylight", "Blackbody")


@APP.callback(
    Output(_uid("advanced-options-collapse"), "is_open"),
    [Input(_uid("toggle-advanced-options-button"), "n_clicks")],
    [State(_uid("advanced-options-collapse"), "is_open")],
)
def toggle_advanced_options(n_clicks, is_open):
    """
    Collapse the *Advanced Options* `Collapse` panel when the
    *Toggle Advanced Options* `Button` is clicked.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    is_open : bool
        Whether the *Advanced Options* `Collapse` panel is opened or collapsed.

    Returns
    -------
    bool
        Whether to open or collapse the *Advanced Options* `Collapse` panel.
    """

    if n_clicks:
        return not is_open

    return is_open


@APP.callback(
    [
        Output(_uid("idt-calculator-output"), "children"),
        Output(_uid("output-data-div"), "children"),
        Output(_uid("idt-calculator-pre"), "style"),
    ],
    [
        Input(_uid("compute-idt-button"), "n_clicks"),
        Input(_uid("formatter-select"), "value"),
        Input(_uid("decimals-select"), "value"),
        Input(_uid("exposure-factor-select"), "value"),
    ],
    [
        State(_uid("camera-sensitivities-select"), "value"),
        State(_uid("camera-sensitivities-datatable"), "data"),
        State(_uid("illuminant-select"), "value"),
        State(_uid("illuminant-datatable"), "data"),
        State(_uid("rgb-display-colourspace-select"), "value"),
        State(_uid("training-data-select"), "value"),
        State(_uid("chromatic-adaptation-transform-select"), "value"),
        State(_uid("optimisation-space-select"), "value"),
        State(_uid("camera-sensitivities-interpolator-select"), "value"),
        State(_uid("illuminant-interpolator-select"), "value"),
        State(_uid("url"), "href"),
    ],
    prevent_initial_call=True,
)
def compute_idt_p2013_001(
    n_clicks,
    formatter,
    decimals,
    exposure_factor,
    camera_name,
    sensitivities_data,
    illuminant_name,
    illuminant_data,
    RGB_display_colourspace,
    training_data,
    chromatic_adaptation_transform,
    optimisation_space,
    sensitivities_interpolator,
    illuminant_interpolator,
    href,
):
    """
    Compute the *Input Device Transform* (IDT).

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    formatter : str
        Formatter to use, :func:`str`, :func:`repr` or *Nuke*.
    decimals : int
        Decimals to use when formatting the IDT matrix.
    exposure_factor : numeric
        Exposure adjustment factor :math:`k` to normalize 18% grey.
    camera_name : str
        Name of the camera.
    sensitivities_data : list
        List of wavelength dicts of camera sensitivities data.
    illuminant_name : str
        Name of the illuminant.
    RGB_display_colourspace : str
        *RGB* display colourspace.
    illuminant_data : list
        List of wavelength dicts of illuminant data.
    training_data : str
        Name of the training data.
    chromatic_adaptation_transform : str
        Name of the chromatic adaptation transform.
    optimisation_space : str
        Name of the optimisation space used to select the corresponding
        optimisation factory.
    sensitivities_interpolator : str
        Name of the camera sensitivities interpolator.
    illuminant_interpolator : str
        Name of the illuminant interpolator.
    href
        URL.

    Returns
    -------
    tuple
        Tuple of *Dash* components.
    """

    key = (
        camera_name,
        hash(
            tuple(
                tuple(
                    [
                        data.get("wavelength"),
                        data.get("R"),
                        data.get("G"),
                        data.get("B"),
                    ]
                )
                for data in sensitivities_data
            )
        ),
        illuminant_name,
        hash(
            tuple(
                tuple(
                    [
                        data.get("wavelength"),
                        data.get("irradiance"),
                    ]
                )
                for data in illuminant_data
            )
        ),
        exposure_factor,
        training_data,
        chromatic_adaptation_transform,
        optimisation_space,
        sensitivities_interpolator,
        illuminant_interpolator,
    )

    (
        M,
        RGB_w,
        XYZ,
        RGB,
        illuminant,
        parsed_sensitivities_data,
        parsed_illuminant_data,
    ) = _CACHE_MATRIX_IDT.get(key, [None] * 7)

    if M is None:
        parsed_sensitivities_data = {}
        for data in sensitivities_data:
            red, green, blue = data.get("R"), data.get("G"), data.get("B")
            if None in (red, green, blue):
                return "Please define all the camera sensitivities values!"

            wavelength = data["wavelength"]
            if wavelength == "...":
                return (
                    "Please define all the camera sensitivities wavelengths!"
                )

            parsed_sensitivities_data[wavelength] = as_float_array(
                [red, green, blue]
            )
        sensitivities = RGB_CameraSensitivities(
            parsed_sensitivities_data,
            interpolator=INTERPOLATORS[sensitivities_interpolator],
        )

        parsed_illuminant_data = {}
        for data in illuminant_data:
            irradiance = data.get("irradiance")
            if irradiance is None:
                return "Please define all the illuminant values!"

            wavelength = data["wavelength"]
            if wavelength == "...":
                return "Please define all the illuminant wavelengths!"

            parsed_illuminant_data[wavelength] = as_float(irradiance)
        illuminant = SpectralDistribution(
            parsed_illuminant_data,
            interpolator=INTERPOLATORS[illuminant_interpolator],
        )

        training_dataset = _TRAINING_DATASETS[training_data]
        optimisation_factory = OPTIMISATION_FACTORIES[optimisation_space]
        chromatic_adaptation_transform = (
            None
            if chromatic_adaptation_transform == "None"
            else chromatic_adaptation_transform
        )
        M, RGB_w, XYZ, RGB = matrix_idt(
            sensitivities=sensitivities,
            illuminant=illuminant,
            training_data=training_dataset,
            optimisation_factory=optimisation_factory,
            chromatic_adaptation_transform=chromatic_adaptation_transform,
            additional_data=True,
        )

        _CACHE_MATRIX_IDT[key] = (
            M,
            RGB_w,
            XYZ,
            RGB,
            illuminant,
            parsed_sensitivities_data,
            parsed_illuminant_data,
        )

    with numpy_print_options(
        formatter={"float": f"{{: 0.{decimals}f}}".format},
        threshold=sys.maxsize,
    ):
        now = datetime.now().strftime("%b %d, %Y %H:%M:%S")

        if formatter == "str":
            output = TEMPLATE_DEFAULT_OUTPUT.format(str(M), str(RGB_w))
        elif formatter == "repr":
            output = TEMPLATE_DEFAULT_OUTPUT.format(repr(M), repr(RGB_w))
        elif formatter == "clf":
            output = format_idt_clf(
                camera_name,
                M,
                RGB_w * exposure_factor,
                {
                    "Application": f"{APP_NAME} - {__version__}",
                    "Url": href,
                    "Date": datetime.now().strftime("%b %d, %Y %H:%M:%S"),
                    "ExposureFactor": exposure_factor,
                    "CameraName": camera_name,
                    "SensitivitiesData": str(parsed_sensitivities_data)
                    .replace("array([", "[")
                    .replace("])", "]"),
                    "IlluminantName": illuminant_name,
                    "IlluminantData": parsed_illuminant_data,
                    "RGBDisplayColourspace": RGB_display_colourspace,
                    "TrainingData": training_data,
                    "ChromaticAdaptationTransform": chromatic_adaptation_transform,
                    "OptimisationSpace": optimisation_space,
                    "SensitivitiesInterpolator": sensitivities_interpolator,
                    "IlluminantInterpolator": illuminant_interpolator,
                },
            )
        elif formatter == "ctl":
            output = TEMPLATE_CTL_MODULE.format(
                matrix=format_matrix_ctl(M, decimals),
                multipliers=format_vector_ctl(RGB_w, decimals),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f"{APP_NAME} - {__version__}",
                url=href,
            )
        elif formatter == "dctl":
            output = TEMPLATE_DCTL_MODULE.format(
                matrix=format_matrix_dctl(M, decimals),
                multipliers=format_vector_dctl(RGB_w, decimals),
                # TODO: Reassess computation with decision on
                # ampas/idt-calculator#26. Ideally, there should not be any
                # math in the GUI besides the computation of the IDT itself.
                b_min=format_float(
                    min(RGB_w[0], min(RGB_w[1], RGB_w[2])), decimals
                ),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f"{APP_NAME} - {__version__}",
                url=href,
            )
        elif formatter == "nuke":
            output = TEMPLATE_NUKE_GROUP.format(
                matrix=format_matrix_nuke(M, decimals),
                multipliers=format_vector_nuke(RGB_w),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f"{APP_NAME} - {__version__}",
                url=href,
                group=slugify(
                    "_".join([camera_name, illuminant_name]).lower()
                ),
            )

    def RGB_working_to_RGB_display(RGB):
        """
        Convert given *RGB* array from the working colourspace to the display
        colourspace.
        """

        return RGB_to_RGB(
            RGB,
            RGB_COLOURSPACE_ACES2065_1,
            RGB_COLOURSPACES[RGB_display_colourspace],
            apply_cctf_encoding=True,
        )

    samples_idt = camera_RGB_to_ACES2065_1(RGB / RGB_w, M, RGB_w)
    samples_reference = XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ACES2065_1)

    compare_colour_checkers_idt_correction = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_idt),
        RGB_working_to_RGB_display(samples_reference),
        _TRAINING_DATASET_TO_COLUMNS[training_data],
    )

    delta_E_idt = error_delta_E(samples_idt, samples_reference)

    return (
        output,
        [
            H3(
                f"IDT (ΔE: {np.median(delta_E_idt):.7f})",
                style={"textAlign": "center"},
            ),
            Img(
                src=(
                    f"data:image/png;base64,{compare_colour_checkers_idt_correction}"
                ),
                style={"width": "100%"},
            ),
        ],
        {"display": "block"},
    )


APP.clientside_callback(
    f"""
    function(n_clicks) {{
        var idtCalculatorOutput = document.getElementById(\
"{_uid('idt-calculator-output')}");
        var content = idtCalculatorOutput.textContent;
        navigator.clipboard.writeText(content).then(function() {{
        }}, function() {{
        }});
        return content;
    }}
    """,
    [Output(component_id=_uid("dev-null"), component_property="children")],
    [Input(_uid("copy-to-clipboard-button"), "n_clicks")],
)
