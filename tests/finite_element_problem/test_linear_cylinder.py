import numpy as np
import pint
import pytest

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.sensor_definition.base_sensor import Sensor
from fenicsxconcrete.sensor_definition.other_sensor import ReactionForceSensorBottom
from fenicsxconcrete.unit_registry import ureg


def simple_setup(p: Parameters, displacement: float, sensor: Sensor, bc_setting: pint.Quantity) -> None:
    parameters = Parameters()  # using the current default values

    parameters["log_level"] = "WARNING" * ureg("")
    parameters["bc_setting"] = bc_setting
    parameters["mesh_density"] = 10 * ureg("")
    parameters = parameters + p

    experiment = CompressionCylinder(parameters)

    problem = LinearElasticity(experiment, parameters)
    problem.add_sensor(sensor)

    problem.experiment.apply_displ_load(displacement)

    problem.solve()  # solving this

    # last measurement
    return problem.sensors[sensor.name].data[-1]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("bc_setting", ["fixed", "free"])
def test_force_response(bc_setting: int, degree: int, dim: str) -> None:
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = dim * ureg("")
    p["bc_setting"] = bc_setting * ureg("")
    p["degree"] = degree * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor, p["bc_setting"])

    result = None
    if dim == 2:
        result = p["E"] * p["radius"] * 2 * displacement / p["height"]
    elif dim == 3:
        result = p["E"] * np.pi * p["radius"] ** 2 * displacement / p["height"]

    assert measured == pytest.approx(result.magnitude, 0.01)


@pytest.mark.parametrize("bc_setting", ["fixed", "free"])
def test_errors_dimensions(bc_setting: str) -> None:
    p = Parameters()  # using the current default values
    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["bc_setting"] = bc_setting * ureg("")
    p["degree"] = 2 * ureg("")

    sensor = ReactionForceSensorBottom()

    with pytest.raises(ValueError):
        p["dim"] = 4 * ureg("")
        measured = simple_setup(p, displacement, sensor, p["bc_setting"])


def test_errors_bc_setting() -> None:
    p = Parameters()  # using the current default values
    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = 3 * ureg("")
    p["degree"] = 2 * ureg("")

    sensor = ReactionForceSensorBottom()

    with pytest.raises(ValueError):
        p["bc_setting"] = "wrong" * ureg("")
        measured = simple_setup(p, displacement, sensor, p["bc_setting"])
