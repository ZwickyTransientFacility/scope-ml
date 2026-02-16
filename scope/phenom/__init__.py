"""
This module contains the Scope phenomenological taxonomy.
"""

__version__ = '1.3.0'
__all__ = ["__version__", "taxonomy", "name", "provenance"]

import os
from os.path import join
import yaml

tax_path = join(os.path.dirname(__file__), 'phenomenological.yaml')

with open(tax_path) as taxonomy_yaml:
    taxonomy = yaml.load(taxonomy_yaml, Loader=yaml.FullLoader)

name = 'SCoPe Phenomenological Taxonomy'
provenance = 'https://github.com/bfhealy/scope-phenomenology'
