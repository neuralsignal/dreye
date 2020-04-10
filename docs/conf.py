# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# from jupyter_sphinx_theme import *
# init_theme()
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'dreye'
copyright = '2020, Matthias Christenson'
author = 'Matthias Christenson'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]
autosummary_generate = True
# autodoc_default_flags = ['members']
numpydoc_show_class_members = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'nature'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/gucky92/dreye"
}
html_context = {
    "github_user": "gucky92",
    "github_repo": "https://github.com/gucky92/dreye",
    "github_version": "master",
    "doc_path": "docs"
}
html_logo = '_static/logo.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Don't add .txt suffix to source files (available for Sphinx >= 1.5):
html_sourcelink_suffix = ''
# Work-around until https://github.com/sphinx-doc/sphinx/issues/4229 is solved:
highlight_language = 'python3'
html_scaled_image_link = False
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# Napoleon settings
napoleon_numpy_docstring = True
master_doc = 'index'

# import recommonmark
# from recommonmark.transform import AutoStructify
#
# def setup(app):
#     app.add_transform(AutoStructify)
