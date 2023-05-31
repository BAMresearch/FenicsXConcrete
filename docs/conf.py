# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FEniCSXConcrete"
copyright = "2023, BAM"
author = "BAM"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  # ["myst_parser", "autodoc2", 'sphinx.ext.napoleon']
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
add_module_names = False
# autodoc2_packages = [
#    "../fenicsxconcrete",
# ]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to examples for the gallery
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "download_all_examples": False,
    "show_signature": False,
    "remove_config_comments": True,
    "filename_pattern": "/",
    "ignore_pattern": r"__init__\.py",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = "furo"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
