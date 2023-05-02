import pytest

from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor


@pytest.mark.parametrize("point_sensor", [DisplacementSensor, StressSensor, StrainSensor])
def test_point_sensor(point_sensor) -> None:
    default_setup, default_parameters = LinearElasticity.default_parameters()

    fem_problem = LinearElasticity(default_setup, default_parameters)

    # define sensors
    sensor_location = [0.0, 0.0, 0.0]
    sensor = point_sensor(sensor_location)

    fem_problem.add_sensor(sensor)

    # check that there is no stored data and test Error
    with pytest.raises(RuntimeError):
        fem_problem.sensors[sensor.name].get_last_entry()

    fem_problem.solve()

    # check that something is stored
    data = fem_problem.sensors[sensor.name].get_last_entry()
    assert data is not None
