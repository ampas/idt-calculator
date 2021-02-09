#!/usr/bin/env python3

import dash
import dash_table
from dash_table import DataTable
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_core_components import Dropdown, Markdown, Slider, Upload, Link, Tab, Tabs, Graph
from dash_html_components import A, Code, Div, H3, H5, Li, Pre, Ul
import sys
import os
import urllib.parse
from flask import Flask
import pandas as pd 

from tabs import tab_1
from tabs import tab_2

import colour
from colour.characterisation import idt_matrix
from colour import SDS_ILLUMINANTS
from colour.colorimetry import sds_and_multi_sds_to_multi_sds
from colour.io import read_sds_from_csv_file

app = dash.Dash()

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    html.H1('IDT Matrix Generator'),
    Tabs(id='tabs-input', value='tab-1', children=[
        Tab(label='Sensitivities Only', value='sens-only'),
        Tab(label='Advanced', value='advanced'),
    ]),
    html.Div(id='tabs-content-output')
])

@app.callback(Output('tabs-content-output', 'children'),
              [Input('tabs-input', 'value')])
def render_content(tab):
    if tab == 'sens-only':
        return tab_1.tab_1_layout
    elif tab == 'advanced':
        return tab_2.tab_2_layout

# Tab 1 callback
@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-datatable', 'data')])
def page_1_datatable(data):
    sensitivities = sds_and_multi_sds_to_multi_sds(data)
    M = idt_matrix(sensitivities, ["D55"])

    with colour.utilities.numpy_print_options(
            formatter={'float': ('{{: 0.{0}f}}').format},
            threshold=sys.maxsize):
            M = str(M)
            return M

# Tab 2 callback
@app.callback(Output('page-2-content', 'children'),
              [Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)