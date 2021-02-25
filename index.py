# -*- coding: utf-8 -*-
"""
Index
=====
"""

from dash.dependencies import Input, Output
from dash_core_components import Link, Location, Markdown
from dash_html_components import A, Div, H3, P

import apps.idt_calculator as app_1
from app import APP, SERVER  # noqa

__author__ = 'Alex Forsythe, Gayle McAdams, Thomas Mansencal'
__copyright__ = ('Copyright (C) 2020-2021 '
                 'Academy of Motion Picture Arts and Sciences')
__license__ = 'Academy of Motion Picture Arts and Sciences License Terms'
__maintainer__ = 'Academy of Motion Picture Arts and Sciences'
__email__ = 'acessupport@oscars.org'
__status__ = 'Production'

__all__ = ['load_app']

APP.layout = Div(
    [Location(id='url', refresh=False),
     Div(id='apps', className='row')])


@APP.callback(Output('apps', 'children'), [Input('url', 'pathname')])
def load_app(app):
    """
    Callback loading given app into the appropriate :class:`Div` class
    instance.

    Parameters
    ----------
    app : unicode
        App path.

    Returns
    -------
    Div
        :class:`Div` class instance of the app layout.
    """

    if app == app_1.APP_PATH:
        return app_1.LAYOUT
    else:
        return Div([
            P([
                'Various A.M.P.A.S. colour science ',
                A('Dash', href='https://dash.plot.ly/', target='_blank'),
                ' apps built on top of \n',
                A('ACES Central',
                  href='https://acescentral.com/',
                  target='_blank'), '.'
            ]),
            H3([
                Link(
                    app_1.APP_NAME, href=app_1.APP_PATH, className='app-link')
            ]),
            Markdown(app_1.APP_DESCRIPTION.replace('This app c', 'C')),
        ])


if __name__ == '__main__':
    APP.run_server(debug=True)
