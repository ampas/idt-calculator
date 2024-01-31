"""
Index
=====
"""

import logging

from dash.dcc import Link, Location, Markdown
from dash.dependencies import Input, Output
from dash.html import H3, A, Div, Img, Main, P
from dash_bootstrap_components import (
    Col,
    Container,
    NavbarSimple,
    NavItem,
    NavLink,
    Row,
)

import apps.idt_calculator_p_2013_001 as app_1
import apps.idt_calculator_prosumer_camera as app_2
from app import APP, SERVER  # noqa: F401

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = ["load_app"]


APP.layout = Container(
    [
        Location(id="url", refresh=False),
        NavbarSimple(
            children=[
                NavItem(
                    NavLink(
                        app_1.APP_NAME_SHORT,
                        href=app_1.APP_PATH,
                    )
                ),
                NavItem(
                    NavLink(
                        app_2.APP_NAME_SHORT,
                        href=app_2.APP_PATH,
                    )
                ),
            ],
            brand=Img(
                id="aces-logo",
                src="/assets/aces-logo.png",
            ),
            brand_href="https://acescentral.com",
            color="primary",
            dark=True,
        ),
        Div(id="toc"),
    ],
    id="apps",
    fluid=True,
)


@APP.callback(Output("toc", "children"), [Input("url", "pathname")])
def load_app(app):
    """
    Load given app into the appropriate :class:`Div` class instance.

    Parameters
    ----------
    app : str
        App path.

    Returns
    -------
    Div
        :class:`Div` class instance of the app layout.
    """

    if app == app_1.APP_PATH:
        return app_1.LAYOUT
    elif app == app_2.APP_PATH:
        return app_2.LAYOUT
    else:
        return Container(
            [
                Main(
                    [
                        Row(
                            [
                                Col(
                                    [
                                        P(
                                            [
                                                "Various A.M.P.A.S. colour science ",
                                                A(
                                                    "Dash",
                                                    href="https://dash.plot.ly/",
                                                    target="_blank",
                                                ),
                                                " apps.",
                                            ]
                                        ),
                                        P(
                                            [
                                                H3(
                                                    [
                                                        Link(
                                                            app_1.APP_NAME_LONG,
                                                            href=app_1.APP_PATH,
                                                        ),
                                                    ]
                                                ),
                                                Markdown(
                                                    app_1.APP_DESCRIPTION.replace(
                                                        "This app c", "C"
                                                    )
                                                ),
                                            ]
                                        ),
                                        P(
                                            [
                                                H3(
                                                    [
                                                        Link(
                                                            app_2.APP_NAME_LONG,
                                                            href=app_2.APP_PATH,
                                                        ),
                                                    ]
                                                ),
                                                Markdown(
                                                    app_2.APP_DESCRIPTION.replace(
                                                        "This app c", "C"
                                                    )
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    APP.run_server(debug=True)
