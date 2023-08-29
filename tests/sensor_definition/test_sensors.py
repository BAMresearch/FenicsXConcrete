import pytest

from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition import (
    DisplacementSensor,
    DOHSensor,
    ReactionForceSensor,
    StrainSensor,
    StressSensor,
    TemperatureSensor,
    YoungsModulusSensor,
)
from fenicsxconcrete.util import ureg


def test_base_sensor() -> None:
    """Testing basic functionality using the displacement sensor as example"""
    default_setup, default_parameters = LinearElasticity.default_parameters()
    fem_problem = LinearElasticity(default_setup, default_parameters)

    # define sensors
    sensor_location = [0.0, 0.0, 0.0]
    sensor = DisplacementSensor(sensor_location)
    fem_problem.add_sensor(sensor)

    fem_problem.solve()  # time = dt: dt_default == 1
    fem_problem.solve()  # time = 2*dt:
    u_sensor = fem_problem.sensors.DisplacementSensor

    # testing get data list
    assert u_sensor.get_data_list().units == pytest.approx(u_sensor.units)
    # testing get time list
    assert u_sensor.get_time_list().magnitude == pytest.approx([fem_problem.p["dt"], 2 * fem_problem.p["dt"]])
    # testing get last data point
    assert u_sensor.get_data_list()[-1].magnitude == pytest.approx(u_sensor.get_last_entry().magnitude)
    # testing get data at time x
    assert u_sensor.get_data_list()[1].magnitude == pytest.approx(
        u_sensor.get_data_at_time(t=2 * fem_problem.p["dt"]).magnitude
    )
    # testing value error for wrong time
    with pytest.raises(ValueError):
        u_sensor.get_data_at_time(t=42)
    # testing set unit
    m_data = u_sensor.get_last_entry()
    u_sensor.set_units("mm")
    mm_data = u_sensor.get_last_entry()
    # check units
    assert u_sensor.get_last_entry().units == ureg.millimeter
    # check magnitude
    assert m_data.magnitude == pytest.approx(mm_data.magnitude / 1000)
    # testing metadata report
    metadata = u_sensor.report_metadata()
    true_metadata = {
        "id": "DisplacementSensor",
        "type": "DisplacementSensor",
        "units": "millimeter",
        "dimensionality": "[length]",
    }
    for key in true_metadata:
        assert key in metadata and true_metadata[key] == metadata[key]


@pytest.mark.parametrize(
    "sensor",
    [
        DisplacementSensor,
        ReactionForceSensor,
        StressSensor,
        StrainSensor,
        DOHSensor,
        YoungsModulusSensor,
        TemperatureSensor,
    ],
)
def test_base_units(sensor) -> None:
    """test that the units defined in base_unit for the sensor are actually base units for this system"""
    dummy_value = 1 * sensor.base_unit()
    assert dummy_value.magnitude == dummy_value.to_base_units().magnitude
