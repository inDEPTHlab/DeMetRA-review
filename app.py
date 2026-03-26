from pathlib import Path
from shiny import App, ui

from definitions.app_ui import (
    overview_page, explore_page, review_page, submit_page,
)

from definitions.app_server import app_server

here = Path(__file__).parent

css_file = here / 'css' / 'custom_styles.css'

app_ui = ui.page_fluid(
    ui.include_css(css_file),
    ui.page_navbar(
        overview_page(),
        explore_page(),
        review_page(),
        submit_page(),
        id="navbar",
        title="DeMetRA : Developmental Methylation Risk Atlas",
    ),
)

app = App(app_ui, app_server, static_assets = Path(__file__).parent / "css")

