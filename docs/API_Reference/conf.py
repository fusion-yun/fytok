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
sys.path.insert(0, '/home/salmon/workspace/fytok/python/')
sys.path.insert(0, '/home/salmon/workspace/fymodule-restricted/python/')
sys.path.insert(0, '/home/salmon/workspace/SpDB/python/')
sys.path.insert(0, '/home/salmon/workspace/freegs')
print(sys.path)
# -- Project information -----------------------------------------------------

project = '《托卡马克集成建模和分析框架》 API Reference'
copyright = '2021, 于治 YUZhi@ipp.ac.cn '
author = '于治 (yuzhi@ipp.ac.cn)'

# The full version, including alpha/beta/rc tags
release = '0.0.1-alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    'sphinx.ext.mathjax',
    # 'sphinx.ext.viewcode',
    # 'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    # 'numpydoc',

    "sphinx.ext.napoleon",
    # "sphinx.ext.imgmath",
    "sphinx.ext.imgconverter",
    "sphinx.ext.todo",
    "sphinx.ext.graphviz",
    "sphinxcontrib.bibtex",
    'sphinxcontrib.plantuml',
    "sphinx_math_dollar",
    # 'matplotlib.sphinxext.only_directives',
    # 'matplotlib.sphinxext.plot_directive',
    # 'matplotlib.sphinxext.ipython_directive',
    # 'matplotlib.sphinxext.ipython_console_highlighting',
    "myst_parser",
    #   "docxsphinx"
]

source_suffix = {
    '.rst': 'restructuredtext',
    #     '.txt': 'markdown',
    '.md': 'markdown',
}

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'

add_module_names = False
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_show_sourcelink = False
imgmath_latex = 'xelatex'
imgmath_latex_preamble = r'''
\usepackage{wasysym}
'''
latex_engine = 'xelatex'
latex_elements = {
    'fontpkg': r'''
\setmainfont[Mapping=tex-text]{Noto Serif CJK SC}
\setsansfont[Mapping=tex-text]{Noto Sans Mono CJK SC}
\setmonofont{Noto Sans Mono CJK SC}
''',
    'preamble': r'''
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
\usepackage{polyglossia}
\setdefaultlanguage[variant=american]{english}
\usepackage{wasysym}
\usepackage{esint}
\usepackage{etoolbox}
\patchcmd{\thebibliography}{\section*{\refname}}{}{}{}

\makeatletter
\renewcommand{\pysigline}[1]{%
                                            
\setlength{\py@argswidth}{\dimexpr\labelwidth+\linewidth\relax}%

\item[{\parbox[t]{\py@argswidth}{\raggedright#1}}]}

 \renewcommand{\pysiglinewithargsret}[1]{%
                                           
 \setlength{\py@argswidth}{\dimexpr\labelwidth+\linewidth\relax}%

 \item[{\parbox[t]{\py@argswidth}{\raggedright#1}}]}
                                            
\makeatother
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}
latex_show_urls = 'footnote'

autodoc_member_order = 'bysource'  # "groupwise"

todo_include_todos = True

image_converter_args = ['-verbose']

plantuml_output_format = 'png'

bibtex_bibfiles = ["FyTok.bib"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
