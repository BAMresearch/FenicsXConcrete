"""
This module contains the default unit registry that is used throughout the codebase.


Attributes:
    ureg (pint.UnitRegistry): The default unit registry.

"""

import pint

ureg = pint.UnitRegistry(cache_folder=":auto:")  # initialize unit registry

# user defined dimensions
ureg.define("[moment] = [force] * [length]")
ureg.define("[stress] = [force] / [length]**2")
ureg.define("GWP = [global_warming_potential] = kg_CO2_eq = kg_C02_equivalent = kg_C02eq")
