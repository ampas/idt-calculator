"""
Input Device Transform (IDT) Calculator - Camera
================================================
"""

import logging
import os
import tempfile
import urllib.parse
import uuid

import colour
import numpy as np
from colour import (
    RGB_COLOURSPACES,
    SDS_ILLUMINANTS,
    RGB_to_RGB,
    SpectralDistribution,
    sd_blackbody,
    sd_CIE_illuminant_D_series,
)
from colour.characterisation import camera_RGB_to_ACES2065_1
from colour.models import RGB_COLOURSPACE_ACES2065_1
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import CACHE_REGISTRY, as_float
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from dash.dcc import Download, Link, Location, Markdown, Tab, Tabs, send_file
from dash.dependencies import Input, Output, State
from dash.html import (
    H2,
    H3,
    A,
    Code,
    Div,
    Footer,
    Img,
    Li,
    Main,
    P,
    Pre,
    Ul,
)

# "Input" is already imported above, to avoid clash, we alias it as "Field".
from dash_bootstrap_components import (
    Button,
    Card,
    CardBody,
    CardHeader,
    Col,
    Collapse,
    Container,
)
from dash_bootstrap_components import Input as Field
from dash_bootstrap_components import (
    InputGroup,
    InputGroupText,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader,
    Row,
    Select,
    Spinner,
    Tooltip,
)
from dash_uploader import Upload, callback, configure_upload

from aces.idt import (
    GENERATORS,
    DirectoryStructure,
    IDTGeneratorApplication,
    IDTGeneratorLogCamera,
    IDTProjectSettings,
    error_delta_E,
    generate_reference_colour_checker,
    hash_file,
    png_compare_colour_checkers,
)
from aces.idt.core.transform_id import generate_idt_urn
from app import APP, SERVER_URL, __version__
from apps.common import (
    COLOUR_ENVIRONMENT,
    CUSTOM_WAVELENGTHS,
    DATATABLE_DECIMALS,
    DELAY_TOOLTIP_DEFAULT,
    INTERPOLATORS,
    OPTIONS_CAT,
    OPTIONS_DISPLAY_COLOURSPACES,
    OPTIONS_ILLUMINANT,
    OPTIONS_INTERPOLATION,
    OPTIONS_OPTIMISATION_SPACES,
    metadata_card_default,
)

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "APP_NAME_LONG",
    "APP_NAME_SHORT",
    "APP_PATH",
    "APP_DESCRIPTION",
    "APP_UID",
    "LAYOUT",
    "set_uploaded_idt_archive_location",
    "toggle_advanced_options",
    "set_illuminant_datable",
    "toggle_options_illuminant",
    "compute_idt_camera",
]

LOGGER = logging.getLogger(__name__)

colour.plotting.colour_style()

APP_NAME_LONG = "Academy Input Device Transform (IDT) Calculator - Camera"
"""
App long name.

APP_NAME_LONG : str
"""

APP_NAME_SHORT = "IDT Calculator - Camera"
"""
App short name.

APP_NAME_SHORT : str
"""

APP_PATH = f"/apps/{__name__.split('.')[-1]}"
"""
App path, i.e. app url.

APP_PATH : str
"""

APP_DESCRIPTION = (
    "This app computes the *Input Device Transform* (IDT) "
    "for a series of *ColorChecker Classic* images captured by a camera."
)
"""
App description.

APP_DESCRIPTION : str
"""

APP_UID = hash(APP_NAME_LONG)
"""
App unique id.

APP_UID : str
"""

_ROOT_UPLOADED_IDT_ARCHIVE = tempfile.gettempdir()

configure_upload(APP, _ROOT_UPLOADED_IDT_ARCHIVE)

_PATH_UPLOADED_IDT_ARCHIVE = None
_HASH_IDT_ARCHIVE = None
_IDT_GENERATOR_APPLICATION = None
_PATH_IDT_ZIP = None

_CACHE_DATA_ARCHIVE_TO_SAMPLES = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DATA_ARCHIVE_TO_SAMPLES"
)

_OPTIONS_DECODING_METHOD = [
    {"label": key, "value": key}
    for key in IDTProjectSettings.decoding_method.metadata.options
]

# Create options for the Select component
GENERATOR_OPTIONS = [{"label": name, "value": name} for name in list(GENERATORS.keys())]


def _uid(id_) -> str:
    """
    Generate a unique id for given id by appending the application *UID*.
    """

    return f"{id_}-{APP_UID}"


_LAYOUT_COLUMN_ILLUMINANT_CHILDREN = [
    InputGroup(
        [
            InputGroupText("Illuminant"),
            Select(
                id=_uid("illuminant-select"),
                options=OPTIONS_ILLUMINANT,
                value=OPTIONS_ILLUMINANT[8]["value"],
            ),
        ],
        className="mb-1",
    ),
    Tooltip(
        "Illuminant used to produce the incident reflectance training and "
        'test datasets. Selecting "Daylight" and "Blackbody" displays a new '
        "input field allowing to define a custom colour temperature. External "
        'tabular data, e.g. from "Excel" or "Google Docs" can be pasted '
        "directly.",
        delay=DELAY_TOOLTIP_DEFAULT,
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
                ),
            ]
        ),
    ),
]

_LAYOUT_COLUMN_SETTINGS_CHILDREN = [
    Div(
        Upload(
            id=_uid("idt-archive-upload"),
            text="Click or drop an IDT Archive here to upload!",
            max_file_size=16384,
            chunk_size=128,
            filetypes=["zip"],
            upload_id=uuid.uuid1(),
            # pause_button=True,
        ),
        style={
            "textAlign": "center",
            "width": "100%",
            "display": "inline-block",
        },
        className="mb-2",
    ),
    Card(
        [
            CardHeader("Options"),
            CardBody(
                [
                    InputGroup(
                        [
                            InputGroupText("Select Generator"),
                            Select(
                                id=_uid("generator-select"),
                                options=GENERATOR_OPTIONS,
                                value=IDTGeneratorLogCamera.GENERATOR_NAME,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        "Select the IDT generator to use.",
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("generator-select"),
                    ),
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
                                        id=_uid("rgb-display-colourspace-select"),
                                        options=OPTIONS_DISPLAY_COLOURSPACES,
                                        value=OPTIONS_DISPLAY_COLOURSPACES[0]["value"],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "RGB colourspace used to display images in "
                                "the app. It does not affect the computations.",
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("rgb-display-colourspace-select"),
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
                                "Chromatic adaptation transform used to "
                                'convert the "ColorChecker Classic" '
                                'reflectances under the "ACES" whitepoint.',
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("chromatic-adaptation-transform-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Optimisation Space"),
                                    Select(
                                        id=_uid("optimisation-space-select"),
                                        options=OPTIONS_OPTIMISATION_SPACES,
                                        value=OPTIONS_OPTIMISATION_SPACES[0]["value"],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Colour model used to compute the error "
                                "during the optimisation process. Recent "
                                'models such as "Oklab" and "JzAzBz" tend to '
                                "produce a lower error.",
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("optimisation-space-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Illuminant Interpolator"),
                                    Select(
                                        id=_uid("illuminant-interpolator-select"),
                                        options=OPTIONS_INTERPOLATION,
                                        value=OPTIONS_INTERPOLATION[1]["value"],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Interpolator used to align the illuminant "
                                "to the working spectral shape, i.e. "
                                "`colour.SpectralShape(360, 830, 1)`",
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("illuminant-interpolator-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Decoding Method"),
                                    Select(
                                        id=_uid("decoding-method-select"),
                                        options=_OPTIONS_DECODING_METHOD,
                                        value=_OPTIONS_DECODING_METHOD[0]["value"],
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Method used to merge the per-channel "
                                '"LUT3x1D" produced during the sampling '
                                'process into a "LUT1D". "Median" computes the'
                                'median of the 3 channels of the "LUT3x1D", '
                                '"Average" computes the average, '
                                '"Per Channel" passes the "LUT3x1D" as is, '
                                'and, "ACES" sums the 3 channels using the '
                                '"ACES" weights.',
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("decoding-method-select"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("EV Range"),
                                    Field(
                                        id=_uid("ev-range-input"),
                                        value="-1 0 1",
                                        placeholder="-1 0 1",
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Range of exposure series used to compute the "
                                '"IDT" matrix: Rather that using a single '
                                "exposure series to compute the matrix, the "
                                "median of given range is used.",
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("ev-range-input"),
                            ),
                            InputGroup(
                                [
                                    InputGroupText("Grey Card Reflectance"),
                                    Field(
                                        id=_uid("grey-card-reflectance"),
                                        value="0.18 0.18 0.18",
                                        placeholder="0.18 0.18 0.18",
                                    ),
                                ],
                                className="mb-1",
                            ),
                            Tooltip(
                                "Optional measure grey card reflectance to "
                                'set the exposure factor "k" that results in '
                                'a nominally "18% gray" object in the scene '
                                "producing ACES values [0.18, 0.18, 0.18].",
                                delay=DELAY_TOOLTIP_DEFAULT,
                                target=_uid("ev-range-input"),
                            ),
                        ],
                        id=_uid("advanced-options-collapse"),
                        className="mb-1",
                    ),
                    InputGroup(
                        [
                            InputGroupText("LUT Size"),
                            Select(
                                id=_uid("lut-size-select"),
                                options=[
                                    {"label": str(2**a), "value": 2**a}
                                    for a in range(10, 17, 1)
                                ],
                                value=1024,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        'Size of the linearisation "LUT1D" (or "LUT3x1D").',
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("lut-size-select"),
                    ),
                    InputGroup(
                        [
                            InputGroupText("LUT Smoothing"),
                            Field(
                                id=_uid("lut-smoothing-input-number"),
                                type="number",
                                min=0,
                                max=256,
                                step=1,
                                value=32,
                            ),
                        ],
                        className="mb-1",
                    ),
                    Tooltip(
                        "Standard deviation of the gaussian convolution "
                        'kernel used when smoothing the linearisation "LUT1D" '
                        '(or "LUT3x1D").',
                        delay=DELAY_TOOLTIP_DEFAULT,
                        target=_uid("lut-smoothing-input-number"),
                    ),
                ]
            ),
        ],
        className="mb-2",
    ),
    metadata_card_default(
        _uid,
        InputGroup(
            [
                InputGroupText("ISO"),
                Field(
                    id=_uid("iso-field"),
                    type="number",
                    value="800",
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            "Camera ISO setting value",
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("iso-field"),
        ),
        InputGroup(
            [
                InputGroupText("Temperature"),
                Field(
                    id=_uid("temperature-field"),
                    type="number",
                    value="5600",
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            "Camera temperature setting value",
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("temperature-field"),
        ),
        InputGroup(
            [
                InputGroupText("Additional Camera Settings"),
                Field(
                    id=_uid("additional-camera-settings-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            "Additional camera settings relevant to the image exposure",
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("additional-camera-settings-field"),
        ),
        InputGroup(
            [
                InputGroupText("Lighting Setup Description"),
                Field(
                    id=_uid("lighting-setup-description-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            "Description of the lighting setup.",
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("lighting-setup-description-field"),
        ),
        InputGroup(
            [
                InputGroupText("Debayering Platform"),
                Field(
                    id=_uid("debayering-platform-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            'Name of the debayering platform, e.g. "Resolve"',
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("debayering-platform"),
        ),
        InputGroup(
            [
                InputGroupText("Debayering Settings"),
                Field(
                    id=_uid("debayering-settings-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            "Debayering platform settings",
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("debayering-settings-platform"),
        ),
        InputGroup(
            [
                InputGroupText("Encoding Colourspace"),
                Field(
                    id=_uid("encoding-colourspace-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            'Encoding colourspace, e.g. "ARRI WideGamut"',
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("encoding-colourspace-platform"),
        ),
        InputGroup(
            [
                InputGroupText("Encoding Transfer Function"),
                Field(
                    id=_uid("encoding-transfer-function-field"),
                    type="text",
                    placeholder="...",
                    debounce=True,
                ),
            ],
            className="mb-1",
        ),
        Tooltip(
            'Encoding transfer function, e.g. "LogC4"',
            delay=DELAY_TOOLTIP_DEFAULT,
            target=_uid("encoding-transfer-function-platform"),
        ),
    ),
]

_LAYOUT_COLUMN_IDT_COMPUTATION_CHILDREN = [
    Card(
        [
            CardHeader("Input Device Transform"),
            CardBody(
                Row(
                    [
                        Col(
                            Button(
                                [
                                    Spinner(
                                        [
                                            Div(id=_uid("loading-div")),
                                            "Compute IDT",
                                        ],
                                        id=_uid("compute-idt-loading"),
                                        show_initially=False,
                                        size="sm",
                                    ),
                                ],
                                id=_uid("compute-idt-button"),
                                style={"width": "100%"},
                            ),
                        ),
                        Col(
                            [
                                Button(
                                    "Download IDT",
                                    id=_uid("download-idt-button"),
                                    style={"width": "100%"},
                                ),
                                Download(id=_uid("download-idt-download")),
                            ],
                            id=_uid("download-idt-column"),
                            style={"display": "none"},
                        ),
                    ]
                ),
            ),
        ],
        id=_uid("compute-idt-card"),
        style={"display": "none"},
    ),
    Row(
        Col(
            Div(id=_uid("output-data-div")),
        ),
    ),
    # Add the modal for displaying error messages
    Modal(
        [
            ModalHeader("Error"),
            ModalBody(id=_uid("modal-body")),
            ModalFooter(Button("Close", id=_uid("close-modal"), className="ml-auto")),
        ],
        id=_uid("error-modal"),
        is_open=False,  # The modal is initially closed
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
        Div([H2([P(APP_NAME_LONG, className="text-center")], id="app-title")]),
        Location(id=_uid("url"), refresh=False),
        Main(
            Tabs(
                [
                    Tab(
                        Row(
                            [
                                Col(
                                    _LAYOUT_COLUMN_ILLUMINANT_CHILDREN,
                                    width=3,
                                ),
                                Col(
                                    [
                                        Row(
                                            Col(
                                                _LAYOUT_COLUMN_SETTINGS_CHILDREN,
                                            ),
                                        ),
                                        Row(
                                            Col(
                                                _LAYOUT_COLUMN_IDT_COMPUTATION_CHILDREN,
                                            ),
                                        ),
                                    ],
                                    width=9,
                                ),
                            ]
                        ),
                        label="Computations",
                        className="mt-3",
                    ),
                    Tab(
                        [
                            Markdown(APP_DESCRIPTION),
                            Markdown(f"{APP_NAME_LONG} - {__version__}"),
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


@callback(
    output=Output(_uid("compute-idt-card"), "style"),
    id=_uid("idt-archive-upload"),
)
def set_uploaded_idt_archive_location(filename):
    """
    Set the uploaded *IDT* archive location.

    Parameters
    ----------
    filename : str
        *IDT* archive filename

    Returns
    -------
    dict
        *CSS* stylesheet for *Dash* components.
    """

    logging.info('Setting uploaded "IDT" archive location to "%s".', filename)

    global _PATH_UPLOADED_IDT_ARCHIVE  # noqa: PLW0603
    global _HASH_IDT_ARCHIVE  # noqa: PLW0603

    _PATH_UPLOADED_IDT_ARCHIVE = filename[0]
    _HASH_IDT_ARCHIVE = None

    return {"display": "block"}


@APP.callback(
    [Output(_uid("acestransformid-field"), "value", allow_duplicate=True)],
    [
        Input(_uid("acesusername-field"), "value"),
        Input(_uid("encoding-colourspace-field"), "value"),
        Input(_uid("encoding-transfer-function-field"), "value"),
    ],
    prevent_initial_call=True,
)
def update_aces_transform_id(
    aces_user_name, encoding_colourspace, encoding_transfer_function
):
    """
    Generate a new ACES transformID based on changes in
    ACES userName, encoding-colourspace, and encoding-transfer-function.

    Parameters
    ----------
    aces_user_name : str
        ACES username.
    encoding_colourspace : str
        Encoding colour space.
    encoding_transfer_function : str
        Encoding transfer function.

    Returns
    -------
    str
        The generated ACEStransformID.
    """

    return [
        generate_idt_urn(
            aces_user_name or "",
            encoding_colourspace or "",
            encoding_transfer_function or "",
            1,
        )
    ]


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

    logging.info("Toggling advanced options...")

    if n_clicks:
        return not is_open

    return is_open


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

    logging.info(
        'Setting illuminant datatable for "%s" illuminant and "%s" CCT...',
        illuminant,
        CCT,
    )

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
            dict(wavelength=wavelength, irradiance=None)
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
                irradiance=illuminant[wavelength],
            )
            for wavelength in illuminant.wavelengths
        ]

    return data, columns


@APP.callback(
    Output(_uid("illuminant-options-collapse"), "is_open"),
    [Input(_uid("illuminant-select"), "value")],
    [State(_uid("illuminant-options-collapse"), "is_open")],
)
def toggle_options_illuminant(illuminant, is_open):  # noqa: ARG001
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

    logging.info("Toggling illuminant options...")

    return illuminant in ("Daylight", "Blackbody")


@APP.callback(
    Output(_uid("download-idt-download"), "data"),
    [Input(_uid("download-idt-button"), "n_clicks")],
    [
        State(_uid("acestransformid-field"), "value"),
        State(_uid("acesusername-field"), "value"),
        State(_uid("camera-make-field"), "value"),
        State(_uid("camera-model-field"), "value"),
        State(_uid("iso-field"), "value"),
        State(_uid("temperature-field"), "value"),
        State(_uid("additional-camera-settings-field"), "value"),
        State(_uid("lighting-setup-description-field"), "value"),
        State(_uid("debayering-platform-field"), "value"),
        State(_uid("debayering-settings-field"), "value"),
        State(_uid("encoding-colourspace-field"), "value"),
        State(_uid("encoding-transfer-function-field"), "value"),
    ],
    prevent_initial_call=True,
)
def download_idt_zip(
    n_clicks,  # noqa: ARG001
    aces_transform_id,
    aces_user_name,
    camera_make,
    camera_model,
    iso,
    temperature,
    additional_camera_settings,
    lighting_setup_description,
    debayering_platform,
    debayering_settings,
    encoding_colourspace,
    encoding_transfer_function,
):
    """
    Download the *IDT* zip file.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    aces_transform_id : str
        *ACEStransformID* of the IDT, e.g.
        *urn:ampas:aces:transformId:v1.5:IDT.ARRI.ARRI-LogC4.a1.v1*.
    aces_user_name : str
        *ACESuserName* of the IDT, e.g. *ACES 1.0 Input - ARRI LogC4*.
    camera_make : str
        Manufacturer of the camera, e.g. *ARRI* or *RED*.
    camera_model : str
        Model of the camera, e.g. *ALEXA 35* or *V-RAPTOR XL 8K VV*.
    iso : float
        Camera ISO setting value.
    temperature : float
        Camera temperature setting value.
    additional_camera_settings : str
        Additional camera settings relevant to the image exposure.
    lighting_setup_description : str
        Description of the lighting setup.
    debayering_platform : str
        Name of the debayering platform, e.g. "Resolve".
    debayering_settings : str
        Debayering platform settings.
    encoding_colourspace : str
        Encoding colourspace, e.g. "ARRI LogC4".

    Returns
    -------
    dict
        Dict of file content (base64 encoded) and metadata used by the
        Download component.
    """

    _IDT_GENERATOR_APPLICATION.project_settings.aces_transform_id = str(
        aces_transform_id
    )
    _IDT_GENERATOR_APPLICATION.project_settings.aces_user_name = str(aces_user_name)
    _IDT_GENERATOR_APPLICATION.project_settings.camera_make = str(camera_make)
    _IDT_GENERATOR_APPLICATION.project_settings.camera_model = str(camera_model)
    _IDT_GENERATOR_APPLICATION.project_settings.iso = float(iso)
    _IDT_GENERATOR_APPLICATION.project_settings.temperature = float(temperature)
    _IDT_GENERATOR_APPLICATION.project_settings.additional_camera_settings = str(
        additional_camera_settings
    )
    _IDT_GENERATOR_APPLICATION.project_settings.lighting_setup_description = str(
        lighting_setup_description
    )
    _IDT_GENERATOR_APPLICATION.project_settings.debayering_platform = str(
        debayering_platform
    )
    _IDT_GENERATOR_APPLICATION.project_settings.debayering_settings = str(
        debayering_settings
    )
    _IDT_GENERATOR_APPLICATION.project_settings.encoding_colourspace = str(
        encoding_colourspace
    )
    _IDT_GENERATOR_APPLICATION.project_settings.encoding_transfer_function = str(
        encoding_transfer_function
    )

    logging.info('Sending "IDT" archive...')

    global _PATH_IDT_ZIP  # noqa: PLW0603

    _PATH_IDT_ZIP = _IDT_GENERATOR_APPLICATION.zip(
        os.path.dirname(_PATH_UPLOADED_IDT_ARCHIVE),
    )

    return send_file(_PATH_IDT_ZIP)


@APP.callback(
    Output(_uid("error-modal"), "is_open"),
    [Input(_uid("close-modal"), "n_clicks")],
    [State(_uid("error-modal"), "is_open")],
)
def toggle_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@APP.callback(
    [
        Output(_uid("loading-div"), "children"),
        Output(_uid("output-data-div"), "children"),
        Output(_uid("download-idt-column"), "style"),
        Output(_uid("acestransformid-field"), "value"),
        Output(_uid("acesusername-field"), "value"),
        Output(_uid("camera-make-field"), "value"),
        Output(_uid("camera-model-field"), "value"),
        Output(_uid("iso-field"), "value"),
        Output(_uid("temperature-field"), "value"),
        Output(_uid("additional-camera-settings-field"), "value"),
        Output(_uid("lighting-setup-description-field"), "value"),
        Output(_uid("debayering-platform-field"), "value"),
        Output(_uid("debayering-settings-field"), "value"),
        Output(_uid("encoding-colourspace-field"), "value"),
        Output(_uid("encoding-transfer-function-field"), "value"),
        Output(_uid("modal-body"), "children"),
        Output(_uid("error-modal"), "is_open", allow_duplicate=True),
    ],
    [
        Input(_uid("compute-idt-button"), "n_clicks"),
    ],
    [
        State(_uid("generator-select"), "value"),
        State(_uid("acestransformid-field"), "value"),
        State(_uid("acesusername-field"), "value"),
        State(_uid("camera-make-field"), "value"),
        State(_uid("camera-model-field"), "value"),
        State(_uid("iso-field"), "value"),
        State(_uid("temperature-field"), "value"),
        State(_uid("additional-camera-settings-field"), "value"),
        State(_uid("lighting-setup-description-field"), "value"),
        State(_uid("debayering-platform-field"), "value"),
        State(_uid("debayering-settings-field"), "value"),
        State(_uid("encoding-colourspace-field"), "value"),
        State(_uid("encoding-transfer-function-field"), "value"),
        State(_uid("rgb-display-colourspace-select"), "value"),
        State(_uid("illuminant-select"), "value"),
        State(_uid("illuminant-datatable"), "data"),
        State(_uid("chromatic-adaptation-transform-select"), "value"),
        State(_uid("optimisation-space-select"), "value"),
        State(_uid("illuminant-interpolator-select"), "value"),
        State(_uid("decoding-method-select"), "value"),
        State(_uid("ev-range-input"), "value"),
        State(_uid("grey-card-reflectance"), "value"),
        State(_uid("lut-size-select"), "value"),
        State(_uid("lut-smoothing-input-number"), "value"),
    ],
    prevent_initial_call=True,
)
def compute_idt_camera(
    n_clicks,  # noqa: ARG001
    generator_name,
    aces_transform_id,
    aces_user_name,
    camera_make,
    camera_model,
    iso,
    temperature,
    additional_camera_settings,
    lighting_setup_description,
    debayering_platform,
    debayering_settings,
    encoding_colourspace,
    encoding_transfer_function,
    RGB_display_colourspace,
    illuminant_name,
    illuminant_data,
    chromatic_adaptation_transform,
    optimisation_space,
    illuminant_interpolator,
    decoding_method,
    EV_range,
    grey_card_reflectance,
    LUT_size,
    LUT_smoothing,
):
    """
    Compute the *Input Device Transform* (IDT) for a camera.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    generator_name : str
        The name of the generator to use.
    aces_transform_id : str
        *ACEStransformID* of the IDT, e.g.
        *urn:ampas:aces:transformId:v1.5:IDT.ARRI.ARRI-LogC4.a1.v1*.
    aces_user_name : str
        *ACESuserName* of the IDT, e.g. *ACES 1.0 Input - ARRI LogC4*.
    camera_make : str
        Manufacturer of the camera, e.g. *ARRI* or *RED*.
    camera_model : str
        Model of the camera, e.g. *ALEXA 35* or *V-RAPTOR XL 8K VV*.
    iso : float
        Camera ISO setting value.
    temperature : float
        Camera temperature setting value.
    additional_camera_settings : str
        Additional camera settings relevant to the image exposure.
    lighting_setup_description : str
        Description of the lighting setup.
    debayering_platform : str
        Name of the debayering platform, e.g. "Resolve".
    debayering_settings : str
        Debayering platform settings.
    encoding_colourspace : str
        Encoding colourspace, e.g. "ARRI LogC4".
    RGB_display_colourspace : str
        *RGB* display colourspace.
    illuminant_name : str
        Name of the illuminant.
    illuminant_data : list
        List of wavelength dicts of illuminant data.
    chromatic_adaptation_transform : str
        Name of the chromatic adaptation transform.
    optimisation_space : str
        Name of the optimisation space used to select the corresponding
        optimisation factory.
    illuminant_interpolator : str
        Name of the illuminant interpolator.
    decoding_method : str
        {"Median", "Average", "Per Channel", "ACES"},
        Decoding method.
    EV_range : str
        Exposure values to use when computing the *IDT* matrix.
    grey_card_reflectance : str
        Measured grey card reflectance.
    LUT_size : integer
        *LUT* size.
    LUT_smoothing : integer
        Standard deviation of the gaussian convolution kernel used for
        smoothing.

    Returns
    -------
    tuple
        Tuple of *Dash* components.
    """

    logging.info(
        'Computing "IDT" with "%s" using parameters:\n'
        '\tRGB Display Colourspace : "%s"\n'
        '\tIlluminant Name : "%s"\n'
        '\tIlluminant Data : "%s"\n'
        '\tChromatic Adaptation Transform : "%s"\n'
        '\tOptimisation Space : "%s"\n'
        '\tIlluminant Interpolator : "%s"\n'
        '\tDecoding Method : "%s"\n'
        '\tEV Range : "%s"\n'
        '\tGrey Card Reflectance : "%s"\n'
        '\tLUT Size : "%s"\n'
        '\tLUT Smoothing : "%s"\n',
        generator_name,
        RGB_display_colourspace,
        illuminant_name,
        illuminant_data,
        chromatic_adaptation_transform,
        optimisation_space,
        illuminant_interpolator,
        decoding_method,
        EV_range,
        grey_card_reflectance,
        LUT_size,
        LUT_smoothing,
    )

    aces_transform_id = str(aces_transform_id)
    aces_user_name = str(aces_user_name or "")
    camera_make = str(camera_make or "")
    camera_model = str(camera_model or "")
    iso = float(iso)
    temperature = float(temperature)
    additional_camera_settings = str(additional_camera_settings)
    lighting_setup_description = str(lighting_setup_description)
    debayering_platform = str(debayering_platform)
    debayering_settings = str(debayering_settings)
    encoding_colourspace = str(encoding_colourspace or "")
    encoding_transfer_function = str(encoding_transfer_function or "")

    # Validation: Check if the inputs are valid
    is_valid, errors = IDTProjectSettings.validate_core_requirements(
        aces_user_name,
        encoding_colourspace,
        encoding_transfer_function,
        camera_make,
        camera_model,
        aces_transform_id,
    )
    if not is_valid:
        error_components = [P(e) for e in errors]
        return (
            "",  # loading-div children
            [],  # output-data-div children
            {"display": "none"},  # download-idt-column style
            aces_transform_id,  # acestransformid-field value
            aces_user_name,  # acesusername-field value
            camera_make,  # camera-make-field value
            camera_model,  # camera-model-field value
            iso,  # iso-field value
            temperature,  # temperature-field value
            additional_camera_settings,  # additional-camera-settings-field value
            lighting_setup_description,  # lighting-setup-description-field value
            debayering_platform,  # debayering-platform-field value
            debayering_settings,  # debayering-settings-field value
            encoding_colourspace,  # encoding-colourspace-field value
            encoding_transfer_function,  # encoding-transfer-function-field value
            error_components,  # Modal body content (error message)
            True,  # Show the modal
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
    chromatic_adaptation_transform = (
        None
        if chromatic_adaptation_transform == "None"
        else chromatic_adaptation_transform
    )

    reference_colour_checker = generate_reference_colour_checker(
        illuminant=illuminant,
        chromatic_adaptation_transform=chromatic_adaptation_transform,
    )

    global _IDT_GENERATOR_APPLICATION  # noqa: PLW0603
    global _HASH_IDT_ARCHIVE  # noqa: PLW0603

    project_settings = IDTProjectSettings(
        aces_transform_id=aces_transform_id,
        aces_user_name=aces_user_name,
        camera_make=camera_make,
        camera_model=camera_model,
        iso=iso,
        temperature=temperature,
        additional_camera_settings=additional_camera_settings,
        lighting_setup_description=lighting_setup_description,
        debayering_platform=debayering_platform,
        debayering_settings=debayering_settings,
        encoding_colourspace=encoding_colourspace,
        encoding_transfer_function=encoding_transfer_function,
        illuminant=illuminant_name,
    )
    _IDT_GENERATOR_APPLICATION = IDTGeneratorApplication(
        generator_name, project_settings
    )

    if _HASH_IDT_ARCHIVE is None:
        _HASH_IDT_ARCHIVE = hash_file(_PATH_UPLOADED_IDT_ARCHIVE)
        LOGGER.debug('"Archive hash: "%s"', _HASH_IDT_ARCHIVE)

    if _CACHE_DATA_ARCHIVE_TO_SAMPLES.get(_HASH_IDT_ARCHIVE) is None:
        _IDT_GENERATOR_APPLICATION.extract(_PATH_UPLOADED_IDT_ARCHIVE)
        os.remove(_PATH_UPLOADED_IDT_ARCHIVE)
        _IDT_GENERATOR_APPLICATION.generator.sample()
        _CACHE_DATA_ARCHIVE_TO_SAMPLES[_HASH_IDT_ARCHIVE] = (
            _IDT_GENERATOR_APPLICATION.project_settings.data,
            _IDT_GENERATOR_APPLICATION.generator.samples_analysis,
            _IDT_GENERATOR_APPLICATION.generator.baseline_exposure,
        )
    else:
        (
            _IDT_GENERATOR_APPLICATION.project_settings.data,
            _IDT_GENERATOR_APPLICATION.generator._samples_analysis,  # noqa: SLF001
            _IDT_GENERATOR_APPLICATION.generator._baseline_exposure,  # noqa: SLF001
        ) = _CACHE_DATA_ARCHIVE_TO_SAMPLES[_HASH_IDT_ARCHIVE]

    generator = _IDT_GENERATOR_APPLICATION.generator
    project_settings = _IDT_GENERATOR_APPLICATION.project_settings

    # TODO: Should really use the application.process to run this, that way we
    # dont have to duplicate the execution logic everywhere however technically
    # nothing wrong with this just means more maintenance.
    generator.sort()
    generator.remove_clipped_samples()
    generator.generate_LUT()
    generator.filter_LUT()
    generator.decode()
    generator.optimise()

    logging.info(str(generator))

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

    samples_median = _IDT_GENERATOR_APPLICATION.generator.samples_analysis[
        DirectoryStructure.COLOUR_CHECKER
    ][generator.baseline_exposure]["samples_median"]

    samples_idt = camera_RGB_to_ACES2065_1(
        # "camera_RGB_to_ACES2065_1" divides RGB by "min(RGB_w)" for highlights
        # recovery, this is not required here as the images are expected to be
        # fully processed, thus we pre-emptively multiply by "min(RGB_w)".
        generator.LUT_decoding.apply(samples_median) * np.min(generator.RGB_w),
        generator.M,
        generator.RGB_w,
        generator.k,
    )

    if generator.baseline_exposure != 0:
        LOGGER.warning(
            "Compensating display and metric computations for non-zero "
            "baseline exposure!"
        )

        samples_idt *= pow(2, -generator.baseline_exposure)

    compare_colour_checkers_idt_correction = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_idt),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    samples_decoded = generator.LUT_decoding.apply(samples_median)
    compare_colour_checkers_LUT_correction = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_decoded),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    compare_colour_checkers_baseline = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_median),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    delta_E_idt = np.median(error_delta_E(samples_idt, reference_colour_checker))
    delta_E_decoded = np.median(
        error_delta_E(samples_decoded, reference_colour_checker)
    )

    # Delta-E
    components = [
        H3(
            f"IDT (ΔE: {delta_E_idt:.7f})",
            style={"textAlign": "center"},
        ),
        Img(
            src=(f"data:image/png;base64,{compare_colour_checkers_idt_correction}"),
            style={"width": "100%"},
        ),
        H3(
            f"LUT1D (ΔE: {delta_E_decoded:.7f})",
            style={"textAlign": "center"},
        ),
        Img(
            src=(f"data:image/png;base64,{compare_colour_checkers_LUT_correction}"),
            style={"width": "100%"},
        ),
        H3("Baseline", style={"textAlign": "center"}),
        Img(
            src=(f"data:image/png;base64,{compare_colour_checkers_baseline}"),
            style={"width": "100%"},
        ),
    ]

    # Segmentation
    colour_checker_segmentation = generator.png_colour_checker_segmentation()
    if colour_checker_segmentation is not None:
        components += [
            H3("Segmentation", style={"textAlign": "center"}),
            Img(
                src=(f"data:image/png;base64,{colour_checker_segmentation}"),
                style={"width": "100%"},
            ),
        ]
    grey_card_sampling = generator.png_grey_card_sampling()
    if grey_card_sampling is not None:
        components += [
            Img(
                src=(f"data:image/png;base64,{grey_card_sampling}"),
                style={"width": "100%"},
            ),
        ]

    # Camera Samples
    measured_camera_samples = generator.png_measured_camera_samples()
    extrapolated_camera_samples = generator.png_extrapolated_camera_samples()
    if None not in (measured_camera_samples, extrapolated_camera_samples):
        components += [
            H3("Measured Camera Samples", style={"textAlign": "center"}),
            Img(
                src=(f"data:image/png;base64,{measured_camera_samples}"),
                style={"width": "100%"},
            ),
            H3("Filtered Camera Samples", style={"textAlign": "center"}),
            Img(
                src=(f"data:image/png;base64,{extrapolated_camera_samples}"),
                style={"width": "100%"},
            ),
        ]

    # Set success components
    return (
        "",  # loading-div children
        components,  # output-data-div children
        {"display": "block"},  # download-idt-column style
        aces_transform_id,  # acestransformid-field value
        aces_user_name,  # acesusername-field value
        camera_make,  # camera-make-field value
        camera_model,  # camera-model-field value
        iso,  # iso-field value
        temperature,  # temperature-field value
        additional_camera_settings,  # additional-camera-settings-field value
        lighting_setup_description,  # lighting-setup-description-field value
        debayering_platform,  # debayering-platform-field value
        debayering_settings,  # debayering-settings-field value
        encoding_colourspace,  # encoding-colourspace-field value
        encoding_transfer_function,  # encoding-transfer-function-field value
        "",  # Clear modal body content (no error)
        False,  # Hide the modal
    )
