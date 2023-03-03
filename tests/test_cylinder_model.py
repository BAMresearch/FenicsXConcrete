import numpy as np

from fenicsxconcrete.sensor_definition.other_sensor import ReactionForceSensorBottom
from fenicsxconcrete.experimental_setup.concrete_cylinder import (
    ConcreteCylinderExperiment,
)
from fenicsxconcrete.finite_element_problem.linear_elastic_material import (
    linear_elasticity,
)
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg

import pytest


def simple_setup(p, displacement, sensor):
    parameters = Parameters()  # using the current default values

    parameters["log_level"] = "WARNING" * ureg("")
    parameters["bc_setting"] = "free" * ureg("")
    parameters["mesh_density"] = 10 * ureg("")

    parameters = parameters + p

    experiment = ConcreteCylinderExperiment(parameters)

    problem = linear_elasticity(experiment, parameters)
    problem.add_sensor(sensor)

    problem.experiment.apply_displ_load(displacement)

    problem.solve()  # solving this

    # last measurement
    return problem.sensors[sensor.name].data[-1]


# testing the linear elastic response
def test_force_response_2D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 6 * ureg("mm")
    p["height"] = 12 * ureg("mm")
    displacement = -3 * ureg("mm")
    p["dim"] = 2 * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor)

    assert measured == pytest.approx(p.E * p.radius * 2 * displacement / p.height)


def test_force_response_3D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 6 * ureg("mm")
    p["height"] = 12 * ureg("mm")
    displacement = -3 * ureg("mm")
    p["dim"] = 3 * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor)

    # due to meshing errors, only aproximate results to be expected. within 1% is good enough
    assert measured == pytest.approx(
        p.E * np.pi * p.radius.magnitude**2 * displacement.magnitude / p.height.magnitude, 0.01
    )


if __name__ == "__main__":
    test_force_response_2D()
    test_force_response_3D()
