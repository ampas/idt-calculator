"""
Input Device Transform (IDT) Calculator - Prosumer Camera
=========================================================
"""

import os.path

import colour
import tempfile
import urllib.parse
import uuid

import numpy as np
from colour import (
    RGB_COLOURSPACES,
    RGB_to_RGB,
    SDS_ILLUMINANTS,
    SpectralDistribution,
    sd_CIE_illuminant_D_series,
    sd_blackbody,
)
from colour.models import RGB_COLOURSPACE_ACES2065_1
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import as_float, as_int_scalar
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from dash.dcc import Download, Link, Markdown, send_file
from dash.dependencies import Input, Output, State
from dash.html import A, Br, Code, Div, Footer, H3, Img, Li, Main, Pre, Ul
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
    Spinner,
    Tab,
    Tabs,
)

# "Input" is already imported above, to avoid clash, we alias it as "Field".
from dash_bootstrap_components import Input as Field
from dash_uploader import Upload, callback, configure_upload

from aces.idt import (
    apply_idt,
    archive_to_idt,
    error_delta_E,
    generate_reference_colour_checker,
    png_colour_checker_segmentation,
    png_compare_colour_checkers,
    png_extrapolated_camera_samples,
    png_grey_card_sampling,
    png_measured_camera_samples,
    zip_idt,
)
from app import APP, SERVER_URL, __version__
from apps.common import (
    COLOUR_ENVIRONMENT,
    CUSTOM_WAVELENGTHS,
    DATATABLE_DECIMALS,
    INTERPOLATORS,
    OPTIMISATION_FACTORIES,
    OPTIONS_CAT,
    OPTIONS_DISPLAY_COLOURSPACES,
    OPTIONS_ILLUMINANT,
    OPTIONS_INTERPOLATION,
    OPTIONS_OPTIMISATION_SPACES,
    STYLE_DATATABLE,
)

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
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
    "set_uploaded_idt_archive_location",
    "toggle_advanced_options",
    "set_illuminant_datable",
    "toggle_options_illuminant",
    "compute_idt_prosumer_camera",
]

colour.plotting.colour_style()

APP_NAME = "Academy Input Device Transform (IDT) Calculator - Prosumer Camera"
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
    "for a series of *ColorChecker Classic* images captured by a prosumer "
    "camera."
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


_ROOT_UPLOADED_IDT_ARCHIVE = tempfile.gettempdir()

configure_upload(APP, _ROOT_UPLOADED_IDT_ARCHIVE)

_PATH_UPLOADED_IDT_ARCHIVE = None
_PATH_IDT_ZIP = None

_OPTIONS_DECODING_METHOD = [
    {"label": key, "value": key}
    for key in ["Median", "Average", "Per Channel", "ACES"]
]


def _uid(id_):
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
    ),
    Div(Br()),
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
                            InputGroup(
                                [
                                    InputGroupText("Decoding Method"),
                                    Select(
                                        id=_uid("decoding-method-select"),
                                        options=_OPTIONS_DECODING_METHOD,
                                        value=_OPTIONS_DECODING_METHOD[0][
                                            "value"
                                        ],
                                    ),
                                ],
                                className="mb-1",
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
                    InputGroup(
                        [
                            InputGroupText("LUT Smoothing"),
                            Field(
                                id=_uid("lut-smoothing-input-number"),
                                type="number",
                                min=0,
                                max=256,
                                step=1,
                                value=16,
                            ),
                        ],
                        className="mb-1",
                    ),
                ]
            ),
        ],
        className="mb-2",
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
                                                _LAYOUT_COLUMN_OPTIONS_CHILDREN,
                                            ),
                                        ),
                                        Row(
                                            Col(
                                                _LAYOUT_COLUMN_IDT_COMPUTATION_CHILDREN,  # noqa
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

    global _PATH_UPLOADED_IDT_ARCHIVE

    _PATH_UPLOADED_IDT_ARCHIVE = filename[0]

    return {"display": "block"}


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
    Output(_uid("download-idt-download"), "data"),
    Input(_uid("download-idt-button"), "n_clicks"),
    prevent_initial_call=True,
)
def download_idt_zip(n_clicks):
    """
    Download the *IDT* zip file.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.

    Returns
    -------
    dict
        Dict of file content (base64 encoded) and meta data used by the
        Download component.
    """

    return send_file(_PATH_IDT_ZIP)


@APP.callback(
    [
        Output(_uid("loading-div"), "children"),
        Output(_uid("output-data-div"), "children"),
        Output(_uid("download-idt-column"), "style"),
    ],
    [
        Input(_uid("compute-idt-button"), "n_clicks"),
    ],
    [
        State(_uid("rgb-display-colourspace-select"), "value"),
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
def compute_idt_prosumer_camera(
    n_clicks,
    RGB_display_colourspace,
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
    Compute the *Input Device Transform* (IDT) for a prosumer camera.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    RGB_display_colourspace : str
        *RGB* display colourspace.
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

    data_archive_to_idt = archive_to_idt(
        _PATH_UPLOADED_IDT_ARCHIVE,
        archive_to_samples_kwargs={},
        sort_samples_kwargs={
            "reference_colour_checker": reference_colour_checker
        },
        generate_LUT3x1D_kwargs={"size": as_int_scalar(LUT_size)},
        filter_LUT3x1D_kwargs={"sigma": as_int_scalar(LUT_smoothing)},
        decode_samples_kwargs={
            "decoding_method": decoding_method,
            "grey_card_reflectance": np.loadtxt([grey_card_reflectance]),
        },
        matrix_idt_kwargs={
            "EV_range": np.loadtxt([EV_range]),
            "training_data": reference_colour_checker,
            "optimisation_factory": OPTIMISATION_FACTORIES[optimisation_space],
        },
        additional_data=True,
    )

    if os.path.exists(_PATH_UPLOADED_IDT_ARCHIVE):
        os.remove(_PATH_UPLOADED_IDT_ARCHIVE)

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

    data_specification_to_samples = (
        data_archive_to_idt.data_archive_to_samples.data_specification_to_samples
    )
    samples_median = data_specification_to_samples.samples_analysis["data"][
        "colour_checker"
    ][0]["samples_median"]

    samples_idt = apply_idt(
        samples_median,
        data_archive_to_idt.data_decode_samples.LUT_decoding,
        data_archive_to_idt.data_matrix_idt.M,
    )
    compare_colour_checkers_idt_correction = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_idt),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    samples_decoded = (
        data_archive_to_idt.data_decode_samples.LUT_decoding.apply(
            samples_median
        )
    )
    compare_colour_checkers_LUT_correction = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_decoded),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    compare_colour_checkers_baseline = png_compare_colour_checkers(
        RGB_working_to_RGB_display(samples_median),
        RGB_working_to_RGB_display(reference_colour_checker),
    )

    delta_E_idt = error_delta_E(samples_idt, reference_colour_checker)
    delta_E_decoded = error_delta_E(samples_decoded, reference_colour_checker)

    global _PATH_IDT_ZIP

    _PATH_IDT_ZIP = zip_idt(
        data_archive_to_idt, os.path.dirname(_PATH_UPLOADED_IDT_ARCHIVE)
    )

    # Delta-E
    components = [
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
        H3(
            f"LUT1D (ΔE: {np.median(delta_E_decoded):.7f})",
            style={"textAlign": "center"},
        ),
        Img(
            src=(
                f"data:image/png;base64,{compare_colour_checkers_LUT_correction}"
            ),
            style={"width": "100%"},
        ),
        H3("Baseline", style={"textAlign": "center"}),
        Img(
            src=(f"data:image/png;base64,{compare_colour_checkers_baseline}"),
            style={"width": "100%"},
        ),
    ]

    # Segmentation
    components += [
        H3("Segmentation", style={"textAlign": "center"}),
        Img(
            src=(
                f"data:image/png;base64,"
                f"{png_colour_checker_segmentation(data_archive_to_idt)}"
            ),
            style={"width": "100%"},
        ),
    ]
    if (
        data_archive_to_idt.data_archive_to_samples.data_specification_to_samples
        is not None
    ):
        components += [
            Img(
                src=(
                    f"data:image/png;base64,"
                    f"{png_grey_card_sampling(data_archive_to_idt)}"
                ),
                style={"width": "100%"},
            ),
        ]

    # Camera Samples
    components += [
        H3("Measured Camera Samples", style={"textAlign": "center"}),
        Img(
            src=(
                f"data:image/png;base64,"
                f"{png_measured_camera_samples(data_archive_to_idt)}"
            ),
            style={"width": "100%"},
        ),
        H3("Filtered Camera Samples", style={"textAlign": "center"}),
        Img(
            src=(
                f"data:image/png;base64,"
                f"{png_extrapolated_camera_samples(data_archive_to_idt)}"
            ),
            style={"width": "100%"},
        ),
    ]

    return (
        "",
        components,
        {"display": "block"},
    )
