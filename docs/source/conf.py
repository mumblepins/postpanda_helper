# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import toml


def get_pyproject():
    curdir = Path(".").absolute()
    while True:
        pyproject = curdir / "pyproject.toml"
        if pyproject.exists():
            break
        if curdir == curdir.parent:
            raise FileNotFoundError("pyproject.toml not found")
        curdir = curdir.parent
    return pyproject


pyproj_path = get_pyproject()
pyproj_toml = toml.load(pyproj_path)

for p in pyproj_toml["tool"]["poetry"]["packages"]:
    sys.path.insert(0, pyproj_path.parent / p.get("from", "") / p["include"])

# -- Project information -----------------------------------------------------

project = pyproj_toml["tool"]["poetry"]["name"]
author = pyproj_toml["tool"]["poetry"]["authors"][0].split("<")[0].strip()
copyright = f"{datetime.now().year}, {author}"  # noqa: pylint: disable=redefined-builtin
# The full version, including alpha/beta/rc tags
release = f"v{pyproj_toml['tool']['poetry']['version']}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[Any] = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "PostPanda Helper",
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://mumblepins.github.io/postpanda_helper",
    # Set the color and the accent color
    "color_primary": "#003865",
    "color_accent": "#78BE21",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/mumblepins/postpanda_helper",
    "repo_name": "PostPanda Helper",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 2,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    "logo_icon": "&#xe4fb",
}
html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

autoclass_content = "both"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
