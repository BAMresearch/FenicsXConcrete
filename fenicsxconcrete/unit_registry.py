import pint
from pathlib import Path

path_to_cache = Path(__file__).parent / '.pint_cache'
ureg = pint.UnitRegistry(cache_folder=path_to_cache)   # initialize unit registry

# user defined dimensions
ureg.define('[moment] = [force] * [length]')
ureg.define('[stress] = [force] / [length]**2')
ureg.define('GWP = [global_warming_potential] = kg_CO2_eq = kg_C02_equivalent = kg_C02eq')

