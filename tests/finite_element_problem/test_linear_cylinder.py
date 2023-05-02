import numpy as np
import pint
import pytest

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.unit_registry import ureg


def simple_setup(
    p: Parameters, displacement: float, bc_setting: pint.Quantity
) -> tuple[float, dict[str, pint.Quantity]]:
    parameters = {}
    parameters["log_level"] = "WARNING" * ureg("")
    parameters["bc_setting"] = bc_setting
    parameters["mesh_density"] = 10 * ureg("")
    parameters["E"] = 1023 * ureg("MPa")
    parameters["nu"] = 0.0 * ureg("")
    parameters["radius"] = 0.006 * ureg("m")
    parameters["height"] = 0.012 * ureg("m")
    parameters["dim"] = 3 * ureg("")
    parameters["bc_setting"] = bc_setting * ureg("")
    parameters["degree"] = 2 * ureg("")

    parameters.update(p)

    experiment = CompressionCylinder(parameters)
    problem = LinearElasticity(experiment, parameters)
    sensor = ReactionForceSensor()
    problem.add_sensor(sensor)
    problem.experiment.apply_displ_load(displacement)

    problem.solve()  # solving this

    # last measurement, parameter dict
    return problem.sensors[sensor.name].get_last_entry().magnitude[-1], problem.parameters


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("bc_setting", ["fixed", "free"])
def test_force_response(bc_setting: int, degree: int, dim: str) -> None:
    p = {}
    p["dim"] = dim * ureg("")
    p["bc_setting"] = bc_setting * ureg("")
    p["degree"] = degree * ureg("")
    displacement = -0.003 * ureg("m")

    measured, fem_p = simple_setup(p, displacement, p["bc_setting"])

    result = None
    if dim == 2:
        result = fem_p["E"] * fem_p["radius"] * 2 * displacement / fem_p["height"]
    elif dim == 3:
        result = fem_p["E"] * np.pi * fem_p["radius"] ** 2 * displacement / fem_p["height"]

    assert measured == pytest.approx(result.magnitude, 0.05)


@pytest.mark.parametrize("bc_setting", ["fixed", "free"])
def test_errors_dimensions(bc_setting: str) -> None:
    p = {}
    displacement = -0.003 * ureg("m")
    p["bc_setting"] = bc_setting * ureg("")
    p["dim"] = 4 * ureg("")

    with pytest.raises(ValueError):
        measured, fem_p = simple_setup(p, displacement, p["bc_setting"])


def test_errors_bc_setting() -> None:
    p = {}
    displacement = -0.003 * ureg("m")
    p["bc_setting"] = "wrong" * ureg("")

    with pytest.raises(ValueError):
        measured, fem_p = simple_setup(p, displacement, p["bc_setting"])
