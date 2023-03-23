import pytest

from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.unit_registry import ureg


@pytest.mark.parametrize(
    "log_level",
    [
        ["DEBUG", False],
        ["INFO", False],
        ["ERROR", False],
        ["WARNING", False],
        ["CRITICAL", False],
        ["some_string_that is_not_implemented", True],
    ],
)
def test_log_levels(log_level: list[str | bool]) -> None:
    """This function tests all implemented log level in the base material init"""

    log_str = log_level[0]
    error_flag = log_level[1]

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    fem_parameters["log_level"] = log_str * ureg("")

    if not error_flag:
        LinearElasticity(default_experiment, fem_parameters)
    else:
        with pytest.raises(ValueError):
            LinearElasticity(default_experiment, fem_parameters)


def test_sensor_error() -> None:
    """This function tests the add sensor function"""

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    problem = LinearElasticity(default_experiment, fem_parameters)

    with pytest.raises(ValueError):
        problem.add_sensor("not a sensor")


def test_sensor_options() -> None:
    """This function tests the function of creating and deleting sensors"""

    # setting up problem
    setup_parameters = CantileverBeam.default_parameters()
    default_setup, fem_parameters = LinearElasticity.default_parameters()

    sensor_location = [setup_parameters["length"].magnitude, 0.0, 0.0]
    sensor = DisplacementSensor([sensor_location])

    # setting up the problem
    experiment = CantileverBeam(setup_parameters)  # Specifies the domain, discretises it and apply Dirichlet BCs
    problem = LinearElasticity(experiment, fem_parameters)

    # check that no sensors yet exist
    assert problem.sensors == {}

    # add sensor
    problem.add_sensor(sensor)

    # check that sensor exists
    assert problem.sensors != {}

    # check that no data is in sensor
    assert problem.sensors[sensor.name].data == []

    # solving and plotting
    problem.solve()

    # check that some data is in sensor
    assert problem.sensors[sensor.name].data != []

    # check cleaning of sensor data
    problem.clean_sensor_data()
    assert problem.sensors[sensor.name].data == []

    # delete sensor
    problem.delete_sensor()
    assert problem.sensors == {}
