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

import os
import sys
sys.path.insert(0, os.path.abspath(r'..'))

# other directories to create documentation
autoapi_dirs = [r'..\codelib']

# -- Project information -----------------------------------------------------

project = 'python_for_the_financial_economist'
copyright = '2021, Johan Stax Jakobsen'
author = 'Johan Stax Jakobsen'

# The full version, including alpha/beta/rc tags
release = '01-09-2021'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints',
              'sphinx.ext.mathjax',
              'sphinx_math_dollar',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.autosectionlabel',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = '.rst'


master_doc = 'index'

autodoc_typehints = 'description'

default_role = 'autolink'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']