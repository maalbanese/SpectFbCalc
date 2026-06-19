# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../spectfbcalc'))

project = 'SpectFbCalc'
copyright = '2026, Deborah Rotoli, Marianna Albanese, Stefano Della Fera and Federico Fabiano'
author = 'Deborah Rotoli, Marianna Albanese, Stefano Della Fera and Federico Fabiano'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',     # To create summary tables for modules/classes/functions/methods
    'sphinx.ext.napoleon',        # From docstrings in Google style
    'sphinx.ext.viewcode',        # Link at source code of documented Python objects
    'myst_parser',                # To use markdown files as source for documentation
    'sphinx_autodoc_typehints'    # To handle type hints in the documentation
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc configuration ------------------------------------------------------
autodoc_member_order = 'bysource' #order of members in the documentation is the same as in the source code

# -- Napoleon configuration ------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_include_init_with_doc = True 
napoleon_use_param = True
napoleon_use_rtype = True

# -- Typehints configuration ------------------------------------------------------
typehints_document_rtype = True
typehints_use_signature_return = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options per importazioni fittizie (Mock Imports) -----------------------
autodoc_mock_imports = [
    'xarray', 
    'dask', 
    'numpy', 
    'pandas', 
    'scipy', 
    'matplotlib', 
    'climtools', 
    'cdo', 
    'psutil', 
    'yaml'
]
