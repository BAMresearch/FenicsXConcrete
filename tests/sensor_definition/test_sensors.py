import copy

import pytest

from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensorBottom


@pytest.mark.parametrize(
    ("sensor", "material_model"),
    [(DisplacementSensor, LinearElasticity), (ReactionForceSensorBottom, LinearElasticity)],
)
def test_default_dictionaries(sensor, material_model) -> None:
    default_setup, default_parameters = material_model.default_parameters()

    fem_problem = material_model(default_setup, default_parameters)

    # define sensors
    sensor_location = [0.0, 0.0, 0.0]
    sensor = DisplacementSensor([sensor_location])

    fem_problem.add_sensor(sensor)

    fem_problem.solve()
