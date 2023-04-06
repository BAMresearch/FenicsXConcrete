import pytest

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.experimental_setup.uniaxial_cube import UniaxialCubeExperiment
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.unit_registry import ureg


def test_reaction_force_sensor() -> None:
    default_setup, default_parameters = LinearElasticity.default_parameters()
    setup = CompressionCylinder(CompressionCylinder.default_parameters())

    fem_problem = LinearElasticity(setup, default_parameters)

    # define sensors
    sensor1 = ReactionForceSensor()
    fem_problem.add_sensor(sensor1)
    sensor2 = ReactionForceSensor(surface=setup.boundary_bottom())
    fem_problem.add_sensor(sensor2)
    sensor3 = ReactionForceSensor(surface=setup.boundary_top(), name="top_sensor")
    fem_problem.add_sensor(sensor3)

    fem_problem.experiment.apply_displ_load(-0.001 * ureg("m"))
    fem_problem.solve()

    # testing default value
    assert (
        fem_problem.sensors.ReactionForceSensor.get_last_entry()
        == fem_problem.sensors.ReactionForceSensor2.get_last_entry()
    ).all()

    # testing top boundary value
    assert fem_problem.sensors.ReactionForceSensor.get_last_entry().magnitude[-1] == pytest.approx(
        -1 * fem_problem.sensors.top_sensor.get_last_entry().magnitude[-1]
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_full_boundary_reaction(dim: int) -> None:
    setup_parameters = UniaxialCubeExperiment.default_parameters()
    setup_parameters["dim"] = dim * ureg("")
    setup_parameters["degree"] = 1 * ureg("")  # TODO: WHY DOES THIS FAIL FOR degree = 2 ??? !!!
    setup_parameters["strain_state"] = "multiaxial" * ureg("")
    cube = UniaxialCubeExperiment(setup_parameters)
    default_setup, fem_parameters = LinearElasticity.default_parameters()
    fem_parameters["nu"] = 0.2 * ureg("")
    fem_problem = LinearElasticity(cube, fem_parameters)

    # define stress sensor
    if dim == 2:
        sensor_location = [0.5, 0.5, 0.0]
    elif dim == 3:
        sensor_location = [0.5, 0.5, 0.5]
    stress_sensor = StressSensor(sensor_location)
    fem_problem.add_sensor(stress_sensor)

    # define reactionforce sensors
    sensor = ReactionForceSensor(surface=cube.boundary_left())
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface=cube.boundary_right())
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface=cube.boundary_top())
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface=cube.boundary_bottom())
    fem_problem.add_sensor(sensor)
    if dim == 3:
        sensor = ReactionForceSensor(surface=cube.boundary_front())
        fem_problem.add_sensor(sensor)
        sensor = ReactionForceSensor(surface=cube.boundary_back())
        fem_problem.add_sensor(sensor)

    fem_problem.experiment.apply_displ_load(0.002 * ureg("m"))
    fem_problem.solve()

    force_left = fem_problem.sensors.ReactionForceSensor.get_last_entry().magnitude[0]
    force_right = fem_problem.sensors.ReactionForceSensor2.get_last_entry().magnitude[0]
    force_top = fem_problem.sensors.ReactionForceSensor3.get_last_entry().magnitude[-1]
    force_bottom = fem_problem.sensors.ReactionForceSensor4.get_last_entry().magnitude[-1]

    # checking opposing forces left-right and top-bottom
    assert force_left == pytest.approx(-1 * force_right)
    assert force_top == pytest.approx(-1 * force_bottom)
    # checking equal forces on sides
    assert force_left == pytest.approx(force_bottom)

    if dim == 3:
        force_front = fem_problem.sensors.ReactionForceSensor5.get_last_entry().magnitude[1]
        force_back = fem_problem.sensors.ReactionForceSensor6.get_last_entry().magnitude[1]

        # checking opposing forces front-back
        assert force_front == pytest.approx(-1 * force_back)
        # checking equal forces left-front
        assert force_left == pytest.approx(force_front)

    # check homogeneous stress state
    stress = fem_problem.sensors.StressSensor.get_last_entry().magnitude
    if dim == 2:
        assert stress[0] == pytest.approx(stress[3])
    if dim == 3:
        assert stress[0] == pytest.approx(stress[4])
        assert stress[0] == pytest.approx(stress[8])
