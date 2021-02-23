#!/usr/bin/env python3

import dash
import dash_html_components as html

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
from dash_table import DataTable

import colour
from colour.colorimetry import sds_and_msds_to_msds

app = dash.Dash(
    __name__
)

app.config['suppress_callback_exceptions'] = True

params = [
    'Wavelength', 'R Sensitivity', 'G Sensitivity', 'B Sensitivity'
]

app.layout = html.Div([
    html.H1('ACES IDT Calculator'),
    html.Div(id='input-options'),
    html.H4('Spec Sens'),
    html.Label('Min Wavelength'), dcc.Input(id='wlMin', value=380, type='number', placeholder='Minimum Wavelength', debounce=True),
    html.Br(),
    html.Label('Max Wavelength'), dcc.Input(id='wlMax', value=780, type='number', placeholder='Maximum Wavelength', debounce=True),
    html.Br(),
    html.Label('Wavelength Increments'), dcc.Input(id='wlInc', value=5, type='number', placeholder='Maximum Increment', debounce=True),
    DataTable(
        style_table={
            'maxHeight': '50ex',
            'overflowY': 'scroll',
            'width': '50%',
            'minWidth': '30%',
        },
        id='sensitivities',
        columns=(
            [{'id': p, 'name': p} for p in params]
        ),
        data=[
            dict(Model=i, **{param: 0 for param in params})
            for i in range(wlMin, wlMax, wlInc)
        ],
        style_cell={'textAlign': 'left',
                    'font_family': 'sans-serif'},
        editable=True
    ),
    # Tabs(id='tabs-input', value='tab-1', children=[
    #     Tab(label='Sensitivities Only', value='sens-only'),
    #     Tab(label='Advanced', value='advanced'),
    # ]),
    html.Div(id='output')
])

# @app.callback(Output('tabs-content-output', 'children'),
#               [Input('tabs-input', 'value')])
# def render_content(tab):
#     if tab == 'sens-only':
#         return tab_1.tab_1_layout
#     elif tab == 'advanced':
#         return tab_2.tab_2_layout
#
# # Tab 1 callback
# @app.callback(dash.dependencies.Output('page-1-content', 'children'),
#               [dash.dependencies.Input('page-1-datatable', 'data')])
# def page_1_datatable(data):
#     sensitivities = sds_and_multi_sds_to_multi_sds(data)
#     M = colour.idt_matrix(sensitivities, ['D55'])
#
#     with colour.utilities.numpy_print_options(
#             formatter={'float': ('{{: 0.{0}f}}').format},
#             threshold=sys.maxsize):
#             M = str(M)
#             return M
#
# # Tab 2 callback
# @app.callback(Output('page-2-content', 'children'),
#               [Input('page-2-radios', 'value')])
# def page_2_radios(value):
#     return 'You have selected '{}''.format(value)

@app.callback(
    Output('output', 'children'),
    [Input('wlMin','value'), Input('wlMax','value'), Input('wlInc','value')]
)
def update_output(wlMin, wlMax, wlInc):
    return u'wlMin {} to wlMin {} in wlInc {} nm'.format(wlMin, wlMax, wlInc)

@app.calback(
    Output('sensitivities', 'children'),
    [Input('wlMin','value'), Input('wlMax','value'), Input('wlInc','value')]
)

def update_sens(wlMin, wlMax, wlInc):
    return range(wlMin, wlMax, wlInc)
)

if __name__ == '__main__':
    app.run_server(debug=True)
