import numpy as np

from fenicsxconcrete.sensor_definition.other_sensor import ReactionForceSensorBottom
from fenicsxconcrete.experimental_setup.concrete_cylinder import ConcreteCylinderExperiment
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg

import pytest


def simple_setup(p, displacement, sensor, bc_setting):
    parameters = Parameters()  # using the current default values

    parameters["log_level"] = "WARNING" * ureg("")
    parameters["bc_setting"] = bc_setting
    parameters["mesh_density"] = 10 * ureg("")

    parameters = parameters + p

    experiment = ConcreteCylinderExperiment(parameters)

    problem = LinearElasticity(experiment, parameters)
    problem.add_sensor(sensor)

    problem.experiment.apply_displ_load(displacement)

    problem.solve()  # solving this

    # last measurement
    return problem.sensors[sensor.name].data[-1]


# testing the linear elastic response
def test_free_force_response_2D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = 2 * ureg("")
    p["bc_setting"] = "free" * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor, p["bc_setting"])

    result = p.E * p.radius * 2 * displacement / p.height
    assert measured == pytest.approx(result.magnitude)

def test_fixed_force_response_2D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = 2 * ureg("")
    p["bc_setting"] = "fixed" * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor, p["bc_setting"])

    result = p.E * p.radius * 2 * displacement / p.height
    assert measured == pytest.approx(result.magnitude)


def test_free_force_response_3D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = 3 * ureg("")
    p["bc_setting"] = "free" * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor, p["bc_setting"])

    # due to meshing errors, only aproximate results to be expected. within 1% is good enough
    result=p.E * np.pi * p.radius**2 * displacement / p.height
    assert measured == pytest.approx(result.magnitude, 0.01)

def test_fixed_force_response_3D():
    p = Parameters()  # using the current default values

    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["dim"] = 3 * ureg("")
    p["bc_setting"] = "fixed" * ureg("")

    sensor = ReactionForceSensorBottom()
    measured = simple_setup(p, displacement, sensor, p["bc_setting"])

    # due to meshing errors, only aproximate results to be expected. within 1% is good enough
    result=p.E * np.pi * p.radius**2 * displacement / p.height
    assert measured == pytest.approx(result.magnitude, 0.01)


def test_default_parameters():

    p = Parameters()  # using the current default values
    p["E"] = 1023 * ureg("MPa")
    p["nu"] = 0.0 * ureg("")
    p["radius"] = 0.006 * ureg("m")
    p["height"] = 0.012 * ureg("m")
    displacement = -0.003 * ureg("m")
    p["bc_setting"] = "fixed" * ureg("")
    p["dim"] = 3 * ureg("")


    sensor = ReactionForceSensorBottom()

    with pytest.raises(ValueError):

        p["dim"] = 4 * ureg("")
        measured = simple_setup(p, displacement, sensor, p["bc_setting"])
    
    with pytest.raises(ValueError):

        p["bc_setting"] = "wrong" * ureg("")
        measured = simple_setup(p, displacement, sensor, p["bc_setting"])

if __name__ == '__main__':
    test_fixed_force_response_2D()
    test_fixed_force_response_3D()
