# -*- coding: utf-8 -*-
"""
Input Device Transform (IDT) Calculator
=======================================
"""

import colour
import urllib.parse
import sys
from datetime import datetime
from colour import (CubicSplineInterpolator, LinearInterpolator,
                    PchipInterpolator, SDS_ILLUMINANTS, SpectralDistribution,
                    SpragueInterpolator)
from colour.characterisation import (RGB_CameraSensitivities,
                                     optimisation_factory_rawtoaces_v1,
                                     optimisation_factory_JzAzBz)
from dash.dependencies import Input, Output, State
from dash_core_components import Link, Markdown
from dash_bootstrap_components import (Button, Card, CardBody, CardHeader, Col,
                                       Collapse, Container, InputGroup,
                                       InputGroupAddon, Row, Select, Tab, Tabs)
# "Input" is already imported above, to avoid clash, we alias it as "Field".
from dash_bootstrap_components import Input as Field
from dash_html_components import A, Code, Div, Footer, H3, Li, Main, Pre, Ul
from dash_table import DataTable
from dash_table.Format import Format, Scheme

from app import APP, SERVER_URL, __version__
from apps.common import (
    CAMERA_SENSITIVITIES_OPTIONS, CAT_OPTIONS, COLOUR_ENVIRONMENT,
    ILLUMINANT_OPTIONS, MSDS_CAMERA_SENSITIVITIES, TEMPLATE_DEFAULT_OUTPUT,
    TEMPLATE_CTL_MODULE, TEMPLATE_DCTL_MODULE, TEMPLATE_NUKE_GROUP,
    TRAINING_DATA_KODAK190PATCHES, format_float, format_matrix_ctl,
    format_vector_nuke, format_vector_ctl, format_matrix_nuke,
    format_matrix_dctl, format_vector_dctl, slugify)

__author__ = 'Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw'
__copyright__ = ('Copyright (C) 2020-2021 '
                 'Academy of Motion Picture Arts and Sciences')
__license__ = 'Academy of Motion Picture Arts and Sciences License Terms'
__maintainer__ = 'Academy of Motion Picture Arts and Sciences'
__email__ = 'acessupport@oscars.org'
__status__ = 'Production'

__all__ = [
    'APP_NAME', 'APP_PATH', 'APP_DESCRIPTION', 'APP_UID', 'LAYOUT',
    'set_camera_sensitivities_datable', 'set_illuminant_datable',
    'toggle_advanced_options', 'compute_idt_matrix'
]

APP_NAME = 'Academy Input Device Transform (IDT) Calculator'
"""
App name.

APP_NAME : unicode
"""

APP_PATH = '/apps/{0}'.format(__name__.split('.')[-1])
"""
App path, i.e. app url.

APP_PATH : unicode
"""

APP_DESCRIPTION = ('This app computes the *Input Device Transform* (IDT) '
                   'for given camera sensitivities and illuminant.')
"""
App description.

APP_DESCRIPTION : unicode
"""

APP_UID = hash(APP_NAME)
"""
App unique id.

APP_UID : unicode
"""

_DATATABLE_DECIMALS = 7

_CUSTOM_WAVELENGTHS = list(range(380, 395, 5)) + ['...']

_TRAINING_DATASET_OPTIONS = [{
    'label': key,
    'value': key
} for key in ['Kodak - 190 Patches', 'ISO 17321-1']]

_TRAINING_DATASETS = {
    'Kodak - 190 Patches':
    TRAINING_DATA_KODAK190PATCHES,
    'ISO 17321-1':
    colour.colorimetry.sds_and_msds_to_msds(
        colour.SDS_COLOURCHECKERS['ISO 17321-1'].values())
}

_OPTIMISATION_SPACE_OPTIONS = [{
    'label': key,
    'value': key
} for key in ['CIE Lab', 'JzAzBz']]

_OPTIMISATION_FACTORIES = {
    'CIE Lab': optimisation_factory_rawtoaces_v1,
    'JzAzBz': optimisation_factory_JzAzBz,
}

_INTERPOLATION_OPTIONS = [{
    'label': key,
    'value': key
} for key in ['Cubic Spline', 'Linear', 'PCHIP', 'Sprague (1880)']]

_INTERPOLATORS = {
    'Cubic Spline': CubicSplineInterpolator,
    'Linear': LinearInterpolator,
    'PCHIP': PchipInterpolator,
    'Sprague (1880)': SpragueInterpolator,
}

_FORMATTER_OPTIONS = [{
    'label': label,
    'value': value
} for label, value in [('Str', 'str'), ('Repr',
                                        'repr'), ('CTL',
                                                  'ctl'), ('DCTL',
                                                           'dctl'), ('Nuke',
                                                                     'nuke')]]

_STYLE_DATATABLE = {
    'header_background_colour': 'rgb(30, 30, 30)',
    'header_colour': 'rgb(220, 220, 220)',
    'cell_background_colour': 'rgb(50, 50, 50)',
    'cell_colour': 'rgb(220, 220, 220)',
}

_IDT_MATRIX_CACHE = {}


def _uid(id_):
    """
    Generates a unique id for given id by appending the application *UID*.
    """

    return f'{id_}-{APP_UID}'


_LAYOUT_COLUMN_CAMERA_SENSITIVITIES_CHILDREN = [
    InputGroup(
        [
            InputGroupAddon('Camera Sensitivities', addon_type='prepend'),
            Select(
                id=_uid('camera-sensitivities'),
                options=CAMERA_SENSITIVITIES_OPTIONS,
                value=CAMERA_SENSITIVITIES_OPTIONS[0]['value']),
        ],
        className='mb-1'),
    Row([
        Col(
            [
                DataTable(
                    id=_uid('camera-sensitivities-datatable'),
                    editable=True,
                    style_as_list_view=True,
                    style_header={
                        'backgroundColor':
                        _STYLE_DATATABLE['header_background_colour']
                    },
                    style_cell={
                        'backgroundColor':
                        _STYLE_DATATABLE['cell_background_colour'],
                        'color':
                        _STYLE_DATATABLE['cell_colour']
                    },
                ),
            ]),
    ])
]

_LAYOUT_COLUMN_ILLUMINANT_CHILDREN = [
    InputGroup(
        [
            InputGroupAddon('Illuminant', addon_type='prepend'),
            Select(
                id=_uid('illuminant'),
                options=ILLUMINANT_OPTIONS,
                value=ILLUMINANT_OPTIONS[0]['value']),
        ],
        className='mb-1'),
    Row([
        Col(
            [
                DataTable(
                    id=_uid('illuminant-datatable'),
                    editable=True,
                    style_as_list_view=True,
                    style_header={
                        'backgroundColor':
                        _STYLE_DATATABLE['header_background_colour']
                    },
                    style_cell={
                        'backgroundColor':
                        _STYLE_DATATABLE['cell_background_colour'],
                        'color':
                        _STYLE_DATATABLE['cell_colour']
                    },
                ),
            ]),
    ])
]

_LAYOUT_COLUMN_OPTIONS_CHILDREN = [
    Card(
        [
            CardHeader('Options'),
            CardBody([
                Button(
                    'Toggle Advanced Options',
                    id=_uid('toggle-advanced-options-button'),
                    className='mb-2'),
                Collapse(
                    [
                        InputGroup(
                            [
                                InputGroupAddon(
                                    'Training Data', addon_type='prepend'),
                                Select(
                                    id=_uid('training-data'),
                                    options=_TRAINING_DATASET_OPTIONS,
                                    value=(_TRAINING_DATASET_OPTIONS[0]
                                           ['value'])),
                            ],
                            className='mb-1'),
                        InputGroup(
                            [
                                InputGroupAddon('CAT', addon_type='prepend'),
                                Select(
                                    id='chromatic-adaptation-transform-{0}'.
                                    format(APP_UID),
                                    options=CAT_OPTIONS,
                                    value=CAT_OPTIONS[3]['value']),
                            ],
                            className='mb-1'),
                        InputGroup(
                            [
                                InputGroupAddon(
                                    'Optimisation Space',
                                    addon_type='prepend'),
                                Select(
                                    id=_uid('optimisation-space'),
                                    options=_OPTIMISATION_SPACE_OPTIONS,
                                    value=_OPTIMISATION_SPACE_OPTIONS[0]
                                    ['value']),
                            ],
                            className='mb-1'),
                        InputGroup(
                            [
                                InputGroupAddon(
                                    'Camera Sensitivities Interpolator',
                                    addon_type='prepend'),
                                Select(
                                    id='camera-sensitivities-interpolator-{0}'.
                                    format(APP_UID),
                                    options=_INTERPOLATION_OPTIONS,
                                    value=_INTERPOLATION_OPTIONS[3]['value']),
                            ],
                            className='mb-1'),
                        InputGroup(
                            [
                                InputGroupAddon(
                                    'Illuminant Interpolator',
                                    addon_type='prepend'),
                                Select(
                                    id='illuminant-interpolator-{0}'.format(
                                        APP_UID),
                                    options=_INTERPOLATION_OPTIONS,
                                    value=_INTERPOLATION_OPTIONS[1]['value']),
                            ],
                            className='mb-1'),
                        InputGroup(
                            [
                                InputGroupAddon(
                                    'Exposure Factor', addon_type='prepend'),
                                Field(
                                    id='exposure-factor-{0}'.format(APP_UID),
                                    type='number',
                                    value=1,
                                ),
                            ],
                            className='mb-1'),
                    ],
                    id=_uid('advanced-options-collapse'),
                    className='mb-1'),
                InputGroup(
                    [
                        InputGroupAddon('Formatter', addon_type='prepend'),
                        Select(
                            id=_uid('formatter'),
                            options=_FORMATTER_OPTIONS,
                            value='str',
                        ),
                    ],
                    className='mb-1'),
                InputGroup(
                    [
                        InputGroupAddon('Decimals', addon_type='prepend'),
                        Select(
                            id=_uid('decimals'),
                            options=[{
                                'label': str(a),
                                'value': a
                            } for a in range(1, 16, 1)],
                            value=10,
                        ),
                    ],
                    className='mb-1'),
            ]),
        ],
        className='mb-2'),
    Card([
        CardHeader('Input Device Transform Matrix'),
        CardBody([
            Button(
                'Compute IDT Matrix',
                id=_uid('compute-idt-matrix-button'),
                className='mb-2'),
            Button(
                'Copy to Clipboard',
                id=_uid('copy-to-clipboard-button'),
                className='ml-2 mb-2'),
            Pre([
                Code(id=_uid('idt-calculator-output'), className='code shell')
            ],
                id=_uid('idt-calculator-pre'),
                className='mt-2'),
        ]),
    ]),
]

_LAYOUT_COLUMN_FOOTER_CHILDREN = [
    Ul([
        Li([Link('Back to index...', href='/', className='app-link')],
           className='list-inline-item'),
        Li([
            A('Permalink',
              href=urllib.parse.urljoin(SERVER_URL, APP_PATH),
              target='_blank')
        ],
           className='list-inline-item'),
        Li([
            A('ACES Central', href='https://acescentral.com/', target='_blank')
        ],
           className='list-inline-item'),
    ],
       className='list-inline mt-3'),
    Div(id=_uid('dev-null'), style={'display': 'none'})
]

LAYOUT = Container([
    H3([Link(APP_NAME, href=APP_PATH)]),
    Main([
        Tabs([
            Tab([
                Row([
                    Col(_LAYOUT_COLUMN_CAMERA_SENSITIVITIES_CHILDREN, width=4),
                    Col(_LAYOUT_COLUMN_ILLUMINANT_CHILDREN, width=3),
                    Col(_LAYOUT_COLUMN_OPTIONS_CHILDREN, width=5),
                ]),
            ],
                label='Computations',
                className='mt-3'),
            Tab([
                Markdown(APP_DESCRIPTION),
                Markdown('{0} - {1}'.format(APP_NAME, __version__)),
                Pre([Code(COLOUR_ENVIRONMENT, className='code shell')]),
            ],
                label='About',
                className='mt-3'),
        ]),
    ]),
    Footer(
        [
            Container(
                [
                    Row([Col(_LAYOUT_COLUMN_FOOTER_CHILDREN)],
                        className='text-center'),
                ],
                fluid=True)
        ],
        className='footer')
])
"""
App layout, i.e. :class:`Container` class instance.

LAYOUT : Div
"""


@APP.callback(
    [
        Output(
            component_id=_uid('camera-sensitivities-datatable'),
            component_property='data'),
        Output(
            component_id=_uid('camera-sensitivities-datatable'),
            component_property='columns')
    ],
    [Input(_uid('camera-sensitivities'), 'value')],
)
def set_camera_sensitivities_datable(camera_sensitivities):
    """
    Sets the *Camera Sensitivities* `DataTable` content for given camera
    sensitivities name.

    Parameters
    ----------
    camera_sensitivities : unicode
        Existing camera sensitivities name or *Custom*.

    Returns
    -------
    tuple
        Tuple of data and columns.
    """

    labels = ['Wavelength', 'Red', 'Green', 'Blue']
    ids = ['wavelength', 'R', 'G', 'B']
    precision = (
        [None
         ] + [Format(precision=_DATATABLE_DECIMALS, scheme=Scheme.fixed)] * 3)
    columns = ([{
        'id': ids[i],
        'name': label,
        'type': 'numeric',
        'format': precision[i]
    } for i, label in enumerate(labels)])

    if camera_sensitivities == 'Custom':
        data = ([
            dict(wavelength=wavelength, **{column: None
                                           for column in labels})
            for wavelength in _CUSTOM_WAVELENGTHS
        ])

    else:
        camera_sensitivities = MSDS_CAMERA_SENSITIVITIES[camera_sensitivities]

        data = ([
            dict(
                wavelength=wavelength,
                **{
                    column: camera_sensitivities.signals[column][wavelength]
                    for column in camera_sensitivities.labels
                }) for wavelength in camera_sensitivities.wavelengths
        ])

    return data, columns


@APP.callback(
    [
        Output(
            component_id=_uid('illuminant-datatable'),
            component_property='data'),
        Output(
            component_id=_uid('illuminant-datatable'),
            component_property='columns')
    ],
    [Input(_uid('illuminant'), 'value')],
)
def set_illuminant_datable(illuminant):
    """
    Sets the *Illuminant* `DataTable` content for given illuminant name.

    Parameters
    ----------
    illuminant : unicode
        Existing illuminant name or *Custom*.

    Returns
    -------
    tuple
        Tuple of data and columns.
    """

    labels = ['Wavelength', 'Value']
    ids = ['wavelength', 'value']
    precision = ([
        None, Format(precision=_DATATABLE_DECIMALS, scheme=Scheme.fixed)
    ])
    columns = ([{
        'id': ids[i],
        'name': label,
        'type': 'numeric',
        'format': precision[i]
    } for i, label in enumerate(labels)])

    if illuminant == 'Custom':
        data = ([
            dict(wavelength=wavelength, **{'value': None})
            for wavelength in _CUSTOM_WAVELENGTHS
        ])
    else:
        illuminant = SDS_ILLUMINANTS[illuminant]

        data = ([
            dict(
                wavelength=wavelength,
                format=Format(
                    precision=_DATATABLE_DECIMALS, scheme=Scheme.fixed),
                **{'value': illuminant[wavelength]})
            for wavelength in illuminant.wavelengths
        ])

    return data, columns


@APP.callback(
    Output(_uid('advanced-options-collapse'), 'is_open'),
    [Input(_uid('toggle-advanced-options-button'), 'n_clicks')],
    [State(_uid('advanced-options-collapse'), 'is_open')],
)
def toggle_advanced_options(n_clicks, is_open):
    """
    Collapses the *Advanced Options* `Collapse` panel when the
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
    Output(
        component_id=_uid('idt-calculator-output'),
        component_property='children'), [
            Input(_uid('compute-idt-matrix-button'), 'n_clicks'),
            Input(_uid('formatter'), 'value'),
            Input(_uid('decimals'), 'value'),
            Input(_uid('exposure-factor'), 'value'),
        ], [
            State(_uid('camera-sensitivities'), 'value'),
            State(_uid('camera-sensitivities-datatable'), 'data'),
            State(_uid('illuminant'), 'value'),
            State(_uid('illuminant-datatable'), 'data'),
            State(_uid('training-data'), 'value'),
            State(_uid('chromatic-adaptation-transform'), 'value'),
            State(_uid('optimisation-space'), 'value'),
            State(_uid('camera-sensitivities-interpolator'), 'value'),
            State(_uid('illuminant-interpolator'), 'value')
        ],
    prevent_initial_call=True)
def compute_idt_matrix(
        n_clicks,
        formatter,
        decimals,
        exposure_factor,
        camera_name,
        sensitivities_data,
        illuminant_name,
        illuminant_data,
        training_data,
        chromatic_adaptation_transform,
        optimisation_space,
        sensitivities_interpolator,
        illuminant_interpolator,
):
    """
    Computes the *Input Device Transform* (IDT) matrix.

    Parameters
    ----------
    n_clicks : int
        Integer that represents that number of times the button has been
        clicked.
    formatter : unicode
        Formatter to use, :func:`str`, :func:`repr` or *Nuke*.
    decimals : int
        Decimals to use when formatting the IDT matrix.
    exposure_factor : numeric
        Exposure adjustment factor :math:`k` to normalize 18% grey.
    camera_name : unicode
        Name of the camera.
    sensitivities_data : list
        List of wavelength dicts of camera sensitivities data.
    illuminant_name : unicode
        Name of the illuminant.
    illuminant_data : list
        List of wavelength dicts of illuminant data.
    training_data : unicode
        Name of the training data.
    chromatic_adaptation_transform : unicode
        Name of the chromatic adaptation transform.
    optimisation_space : unicode
        Name of the optimisation space used to select the correspond
        optimisation factory.
    sensitivities_interpolator : unicode
        Name of the camera sensitivities interpolator.
    illuminant_interpolator : unicode
        Name of the illuminant interpolator.

    Returns
    -------
    unicode
        IDT matrix.
    """

    key = (
        camera_name,
        hash(
            tuple([
                tuple([
                    data.get('wavelength'),
                    data.get('R'),
                    data.get('G'),
                    data.get('B'),
                ]) for data in sensitivities_data
            ])),
        illuminant_name,
        hash(
            tuple([
                tuple([
                    data.get('wavelength'),
                    data.get('value'),
                ]) for data in illuminant_data
            ])),
        exposure_factor,
        training_data,
        chromatic_adaptation_transform,
        optimisation_space,
        sensitivities_interpolator,
        illuminant_interpolator,
    )

    M, RGB_w, XYZ, RGB = _IDT_MATRIX_CACHE.get(key, [None] * 4)

    if M is None:
        parsed_sensitivities_data = {}
        for data in sensitivities_data:
            red, green, blue = data.get('R'), data.get('G'), data.get('B')
            if None in (red, green, blue):
                return 'Please define all the camera sensitivities values!'

            wavelength = data['wavelength']
            if wavelength == '...':
                return (
                    'Please define all the camera sensitivities wavelengths!')

            parsed_sensitivities_data[wavelength] = (
                colour.utilities.as_float_array([red, green, blue]))
        sensitivities = RGB_CameraSensitivities(
            parsed_sensitivities_data,
            interpolator=_INTERPOLATORS[sensitivities_interpolator])

        parsed_illuminant_data = {}
        for data in illuminant_data:
            value = data.get('value')
            if value is None:
                return 'Please define all the illuminant values!'

            wavelength = data['wavelength']
            if wavelength == '...':
                return 'Please define all the illuminant wavelengths!'

            parsed_illuminant_data[wavelength] = (
                colour.utilities.as_float(value))
        illuminant = SpectralDistribution(
            parsed_illuminant_data,
            interpolator=_INTERPOLATORS[illuminant_interpolator])

        training_data = _TRAINING_DATASETS[training_data]
        optimisation_factory = _OPTIMISATION_FACTORIES[optimisation_space]

        M, RGB_w, XYZ, RGB = colour.matrix_idt(
            sensitivities=sensitivities,
            illuminant=illuminant,
            training_data=training_data,
            optimisation_factory=optimisation_factory,
            chromatic_adaptation_transform=chromatic_adaptation_transform,
            additional_data=True,
        )

        _IDT_MATRIX_CACHE[key] = M, RGB_w, XYZ, RGB

    with colour.utilities.numpy_print_options(
            formatter={'float': ('{{: 0.{0}f}}'.format(decimals)).format},
            threshold=sys.maxsize):

        now = datetime.now().strftime("%b %d, %Y %H:%M:%S")

        if formatter == 'str':
            output = TEMPLATE_DEFAULT_OUTPUT.format(str(M), str(RGB_w))
        elif formatter == 'repr':
            output = TEMPLATE_DEFAULT_OUTPUT.format(repr(M), repr(RGB_w))
        elif formatter == 'ctl':
            output = TEMPLATE_CTL_MODULE.format(
                matrix=format_matrix_ctl(M, decimals),
                multipliers=format_vector_ctl(RGB_w, decimals),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f'{APP_NAME} - {__version__}',
                url=APP_PATH)
        elif formatter == 'dctl':
            output = TEMPLATE_DCTL_MODULE.format(
                matrix=format_matrix_dctl(M, decimals),
                multipliers=format_vector_dctl(RGB_w, decimals),
                # TODO: Reassess computation with decision on
                #  ampas/idt-calculator#26. Ideally, there should not be any
                # math in the GUI besides the computation of the IDT itself.
                b_min=format_float(
                    min(RGB_w[0], min(RGB_w[1], RGB_w[2])), decimals),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f'{APP_NAME} - {__version__}',
                url=APP_PATH)
        elif formatter == 'nuke':
            output = TEMPLATE_NUKE_GROUP.format(
                matrix=format_matrix_nuke(M, decimals),
                multipliers=format_vector_nuke(RGB_w),
                k_factor=format_float(exposure_factor, decimals),
                camera=camera_name,
                illuminant=illuminant_name,
                date=now,
                application=f'{APP_NAME} - {__version__}',
                url=APP_PATH,
                group=slugify('_'.join([camera_name, illuminant_name])))

        return output


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
    [Output(component_id=_uid('dev-null'), component_property='children')],
    [Input(_uid('copy-to-clipboard-button'), 'n_clicks')],
)
