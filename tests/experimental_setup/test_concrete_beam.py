import numpy as np
import fenicsxconcrete
import pytest

def test_simple_beam():
    print('Testing')
    setup = fenicsxconcrete.experimental_setup.simple_beam()
