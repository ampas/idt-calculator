"""
Index
=====
"""

from dash.dcc import Link, Location, Markdown
from dash.dependencies import Input, Output
from dash.html import A, Div, H3, Main, P
from dash_bootstrap_components import Col, Container, Row

import apps.idt_calculator as app_1
import apps.idt_calculator_blackbox_camera as app_2
from app import APP, SERVER  # noqa

__author__ = "Alex Forsythe, Gayle McAdams, Thomas Mansencal, Nick Shaw"
__copyright__ = "Copyright 2020 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = ["load_app"]

APP.layout = Container([Location(id="url", refresh=False), Div(id="apps")])


@APP.callback(Output("apps", "children"), [Input("url", "pathname")])
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
                                                            app_1.APP_NAME,
                                                            href=app_1.APP_PATH,
                                                            className="app-link",
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
                                                            app_2.APP_NAME,
                                                            href=app_2.APP_PATH,
                                                            className="app-link",
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
    APP.run_server(debug=True)
