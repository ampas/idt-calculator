#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components import H5, Pre, Code
import dash_table
from dash_table import DataTable
from dash.dependencies import Input, Output
import pandas as pd

import colour
from colour.colorimetry import sds_and_multi_sds_to_multi_sds

params = [
    'Wavelength', 'R Sensitivity', 'G Sensitivity', 'B Sensitivity'
]

tab_1_layout = html.Div([
    H5(children='Camera System Spectral Sensitivities (Copy & Paste)'), 
        DataTable(
            id='sensitivities',
            columns=(
                [{'id': p, 'name': p} for p in params]
            ),
            data=[
            dict(Model=i, **{param: 0 for param in params})
            for i in range(1, 5)
            ],
            editable=True
        ),
        Pre([Code(id='idt-matrix', className='code shell')],
            className='app-widget app-output'),
])