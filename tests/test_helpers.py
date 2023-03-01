from fenicsxconcrete.helpers import Parameters
from pint import UnitRegistry
ureg = UnitRegistry()

def test_parameters():
    parameters = Parameters()
    parameters["length"] = 42.0 * ureg.cm

    # Check if units are converted correctly
    assert parameters["length"].units == ureg.meter

    # Check if dot access works
    assert parameters.length.units == ureg.meter

    # Check if dot access changes the quantity correctly
    parameters.length = 42.0 * ureg.decimeter
    assert parameters.length.units == ureg.meter

    parameters_2 = Parameters()
    parameters_2["temperature"] = 2.0 * ureg.kelvin

    parameters_combined = parameters + parameters_2
    keys = parameters_combined.keys()
    assert "length" in keys and "temperature" in keys
    assert (
        parameters_combined["length"] == parameters["length"]
        and parameters_combined["temperature"] == parameters_2["temperature"]
    )