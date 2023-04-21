import pytest

from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.unit_registry import ureg


def test_base_sensor() -> None:
    """Testing basic functionality using the displacement sensor as example"""
    default_setup, default_parameters = LinearElasticity.default_parameters()
    fem_problem = LinearElasticity(default_setup, default_parameters)

    # define sensors
    sensor_location = [0.0, 0.0, 0.0]
    sensor = DisplacementSensor(sensor_location)
    fem_problem.add_sensor(sensor)

    fem_problem.solve(t=0.5)
    fem_problem.solve(t=1)
    u_sensor = fem_problem.sensors.DisplacementSensor

    # testing get data list
    assert u_sensor.get_data_list().units == u_sensor.units
    # testing get time list
    assert (u_sensor.get_time_list().magnitude == [0.5, 1]).all()
    # testing get last data point
    assert (u_sensor.get_data_list()[-1] == u_sensor.get_last_entry()).all()
    # testing get data at time x
    assert (u_sensor.get_data_list()[1] == u_sensor.get_data_at_time(t=1)).all()
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
    assert (m_data.magnitude == mm_data.magnitude / 1000).all()


@pytest.mark.parametrize("sensor", [DisplacementSensor, ReactionForceSensor, StressSensor, StrainSensor])
def test_base_units(sensor) -> None:
    """test that the units defined in base_unit for the sensor are actually base units for this system"""
    dummy_value = 1 * sensor.base_unit()
    assert dummy_value.magnitude == dummy_value.to_base_units().magnitude
