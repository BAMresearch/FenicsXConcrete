import pytest

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.experimental_setup.simple_cube import SimpleCube
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


def test_reaction_force_sensor() -> None:
    default_setup, default_parameters = LinearElasticity.default_parameters()
    setup = CompressionCylinder(CompressionCylinder.default_parameters())

    fem_problem = LinearElasticity(setup, default_parameters)

    # define sensors
    sensor1 = ReactionForceSensor()
    fem_problem.add_sensor(sensor1)
    sensor2 = ReactionForceSensor(surface={"function": "boundary_bottom", "args": {}})
    fem_problem.add_sensor(sensor2)
    sensor3 = ReactionForceSensor(surface={"function": "boundary_top", "args": {}}, name="top_sensor")
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
@pytest.mark.parametrize("degree", [1, 2])
def test_full_boundary_reaction(dim: int, degree: int) -> None:
    setup_parameters = SimpleCube.default_parameters()
    setup_parameters["dim"] = dim * ureg("")
    setup_parameters["degree"] = degree * ureg("")
    setup_parameters["strain_state"] = "multiaxial" * ureg("")
    cube = SimpleCube(setup_parameters)
    default_setup, fem_parameters = LinearElasticity.default_parameters()
    fem_parameters["nu"] = 0.2 * ureg("")
    fem_problem = LinearElasticity(cube, fem_parameters)

    # define reactionforce sensors
    sensor = ReactionForceSensor(surface={"function": "boundary_left", "args": {}}, name="ReactionForceSensorLeft")
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface={"function": "boundary_right", "args": {}}, name="ReactionForceSensorRight")
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface={"function": "boundary_top", "args": {}}, name="ReactionForceSensorTop")
    fem_problem.add_sensor(sensor)
    sensor = ReactionForceSensor(surface={"function": "boundary_bottom", "args": {}}, name="ReactionForceSensorBottom")
    fem_problem.add_sensor(sensor)
    if dim == 3:
        sensor = ReactionForceSensor(
            surface={"function": "boundary_front", "args": {}}, name="ReactionForceSensorFront"
        )
        fem_problem.add_sensor(sensor)
        sensor = ReactionForceSensor(surface={"function": "boundary_back", "args": {}}, name="ReactionForceSensorBack")
        fem_problem.add_sensor(sensor)

    fem_problem.experiment.apply_displ_load(0.002 * ureg("m"))
    fem_problem.solve()

    force_left = fem_problem.sensors.ReactionForceSensorLeft.get_last_entry().magnitude[0]
    force_right = fem_problem.sensors.ReactionForceSensorRight.get_last_entry().magnitude[0]
    force_top = fem_problem.sensors.ReactionForceSensorTop.get_last_entry().magnitude[-1]
    force_bottom = fem_problem.sensors.ReactionForceSensorBottom.get_last_entry().magnitude[-1]

    # checking opposing forces left-right and top-bottom
    assert force_left == pytest.approx(-1 * force_right)
    assert force_top == pytest.approx(-1 * force_bottom)
    # checking equal forces on sides
    assert force_left == pytest.approx(force_bottom)
    # checking report metadata
    # TODO Figure out how to identify which boundary is applied
    assert fem_problem.sensors.ReactionForceSensorLeft.report_metadata()["surface"] == {
        "function": "boundary_left",
        "args": {},
    }
    assert fem_problem.sensors.ReactionForceSensorRight.report_metadata()["surface"] == {
        "function": "boundary_right",
        "args": {},
    }
    assert fem_problem.sensors.ReactionForceSensorTop.report_metadata()["surface"] == {
        "function": "boundary_top",
        "args": {},
    }
    assert fem_problem.sensors.ReactionForceSensorBottom.report_metadata()["surface"] == {
        "function": "boundary_bottom",
        "args": {},
    }

    if dim == 3:
        force_front = fem_problem.sensors.ReactionForceSensorFront.get_last_entry().magnitude[1]
        force_back = fem_problem.sensors.ReactionForceSensorBack.get_last_entry().magnitude[1]

        # checking opposing forces front-back
        assert force_front == pytest.approx(-1 * force_back)
        # checking equal forces left-front
        assert force_left == pytest.approx(force_front)
        # checking report metadata
        assert fem_problem.sensors.ReactionForceSensorFront.report_metadata()["surface"] == {
            "function": "boundary_front",
            "args": {},
        }
        assert fem_problem.sensors.ReactionForceSensorBack.report_metadata()["surface"] == {
            "function": "boundary_back",
            "args": {},
        }


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_full_boundary_stress(dim: int, degree: int) -> None:
    setup_parameters = SimpleCube.default_parameters()
    setup_parameters["dim"] = dim * ureg("")
    setup_parameters["degree"] = degree * ureg("")
    setup_parameters["strain_state"] = "multiaxial" * ureg("")
    cube = SimpleCube(setup_parameters)
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

    fem_problem.experiment.apply_displ_load(0.002 * ureg("m"))
    fem_problem.solve()

    # check homogeneous stress state
    stress = fem_problem.sensors.StressSensor.get_last_entry().magnitude
    if dim == 2:
        assert stress[0] == pytest.approx(stress[3])
    if dim == 3:
        assert stress[0] == pytest.approx(stress[4])
        assert stress[0] == pytest.approx(stress[8])
