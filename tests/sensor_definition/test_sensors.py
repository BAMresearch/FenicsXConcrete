import pytest

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
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
    assert (u_sensor.get_data_list()[-1] == u_sensor.get_last_data_point()).all()
    # testing get data at time x
    assert (u_sensor.get_data_list()[1] == u_sensor.get_data_at_time(t=1)).all()
    # testing value error for wrong time
    with pytest.raises(ValueError):
        u_sensor.get_data_at_time(t=42)
    # testing set unit
    m_data = u_sensor.get_last_data_point()
    u_sensor.set_units("mm")
    mm_data = u_sensor.get_last_data_point()
    # check units
    assert u_sensor.get_last_data_point().units == ureg.millimeter
    # check magnitude
    assert (m_data.magnitude == mm_data.magnitude / 1000).all()


@pytest.mark.parametrize("sensor", [DisplacementSensor, ReactionForceSensor, StressSensor, StrainSensor])
def test_base_units(sensor) -> None:
    """test that the units defined in base_unit for the sensor are actually base units for this system"""
    dummy_value = 1 * sensor.base_unit()
    assert dummy_value.magnitude == dummy_value.to_base_units().magnitude


def test_displacement_sensor() -> None:
    default_setup, default_parameters = LinearElasticity.default_parameters()

    fem_problem = LinearElasticity(default_setup, default_parameters)

    # define sensors
    sensor_location = [0.0, 0.0, 0.0]
    sensor = DisplacementSensor(sensor_location)

    fem_problem.add_sensor(sensor)

    fem_problem.solve()


def test_reaction_force_sensor() -> None:
    default_setup, default_parameters = LinearElasticity.default_parameters()
    setup = CompressionCylinder(CompressionCylinder.default_parameters())

    fem_problem = LinearElasticity(setup, default_parameters)

    # define sensors
    sensor1 = ReactionForceSensor()
    fem_problem.add_sensor(sensor1)
    sensor2 = ReactionForceSensor(surface=setup.boundary_bottom())
    fem_problem.add_sensor(sensor2)
    sensor3 = ReactionForceSensor(surface=setup.boundary_top())
    fem_problem.add_sensor(sensor3)

    fem_problem.experiment.apply_displ_load(-0.001 * ureg("m"))
    fem_problem.solve()

    # testing default value
    assert (
        fem_problem.sensors.ReactionForceSensor.get_last_data_point()
        == fem_problem.sensors.ReactionForceSensor2.get_last_data_point()
    )

    # testing top boundary value
    assert fem_problem.sensors.ReactionForceSensor.get_last_data_point().magnitude == pytest.approx(
        -1 * fem_problem.sensors.ReactionForceSensor3.get_last_data_point().magnitude
    )
