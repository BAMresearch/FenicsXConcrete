import numpy as np
from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeamExperiment
import pytest
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_simple_beam(dim):
    # TODO: this simple beam is still a work in progress and will be dealt with in a separate issue
    print('Testing')

    p = Parameters()
    p['length'] = 5 * ureg('m')
    p['height'] = 1 * ureg('m')
    p['width'] = 0.8 * ureg('m')
    p['dim'] = dim * ureg('')

    setup = SimpleBeamExperiment()
    print(dim, setup.mesh)