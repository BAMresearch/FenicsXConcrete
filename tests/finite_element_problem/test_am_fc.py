import os
from pathlib import Path
from typing import Literal

import dolfinx as df
import numpy as np
import pytest


from fenicsxconcrete.experimental_setup import AmMultipleLayers
from fenicsxconcrete.finite_element_problem.concrete_am_fc import ConcreteAMFC
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, ureg

#############Material laws using fenics-constitutive interface for am with changable parameters over time by solver#############
from fenicsxconcrete.finite_element_problem.material_for_am_fc import LinearElasticityModel, VonMises3D

def set_test_parameters(mat: Literal["linear_elastic", "mises"]) -> Parameters:
    """set up a test parameter set

    Args:
        dim: dimension of problem

    Returns: filled instance of Parameters

    """
    setup_parameters = {}

    setup_parameters["dim"] = 3 * ureg("")
    # setup_parameters["stress_state"] = "plane_strain"
    setup_parameters["num_layers"] = 5 * ureg("")  
    setup_parameters["layer_height"] = 1 / 100 * ureg("m")  # y (2D), z (3D)
    setup_parameters["layer_length"] = 50 / 100 * ureg("m")  # x
    setup_parameters["layer_width"] = 3 / 100 * ureg("m")  # y (3D)

    setup_parameters["num_elements_layer_length"] = 4 * ureg("")
    setup_parameters["num_elements_layer_height"] = 2 * ureg("")
    setup_parameters["num_elements_layer_width"] = 4 * ureg("")

    # keep the problem isoparametric: the AmMultipleLayers mesh is first order
    # (geometry degree 1), so use a first-order displacement field. This matches
    # the requirement of fenics-constitutive's corotational mesh update.
    setup_parameters["degree"] = 1 * ureg("")
    setup_parameters["q_degree"] = 4 * ureg("")
    setup_parameters["rho"] = 2000 * ureg("kg/m^3")

    if mat == "linear_elastic":
        material_law = LinearElasticityModel
        setup_parameters["E"] = 42000 * ureg("Pa")  # young's modulus
        setup_parameters["nu"] = 0.3 * ureg("")  # poisson ratio
        setup_parameters["A_E"] = 4000 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
    elif mat == "mises":
        material_law = VonMises3D
        setup_parameters["p_ka"] = 17500.0 * ureg("Pa")  # bulk modulus
        setup_parameters["p_mu"] = 8076.9 * ureg("Pa")  # shear modulus
        setup_parameters["p_y0"] = 1000 * ureg("Pa")  # initial yield stress
        setup_parameters["p_y00"] = 2000 * ureg("Pa")  # final yield stress
        setup_parameters["p_w"] = 0.3 * ureg("")  # saturation parameter
        setup_parameters["A_p_ka"] = 400 * ureg("Pa/s")  # bulk modulus rate over time
        setup_parameters["A_p_mu"] = 400 * ureg("Pa/s")  # shear modulus rate over time
        setup_parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
    else:
        raise ValueError("material not supported")

    return setup_parameters, material_law


@pytest.mark.parametrize("mat", ["linear_elastic", "mises"])
@pytest.mark.parametrize("factor", [1, 2])
def test_am_multiple_layer(
    mat: Literal["linear_elastic", "mises"], factor: int, plot: bool = False
) -> None:
    """multiple layer test

    several layers building over time one layer at once

    Args:
        mat: material law string in the moment only linear elastic
        factor: length of load_time = factor * dt

    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = "test_am_multiple_layer"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters, material_law = set_test_parameters(mat)

    # solving parameters
    time_layer = 10 * ureg("s")  # time to build one layer
    time_steps_layers = 5
    setup_parameters["dt"] = time_layer / time_steps_layers
    assert factor < time_layer / setup_parameters["dt"], "factor is bigger then time for on layer"
    setup_parameters["load_time"] = factor * setup_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)

    problem = ConcreteAMFC(experiment, setup_parameters, material_law, pv_name=file_name, pv_path=data_path)

    # initial path function describing layer activation #TDOD: fix problem in define_path fct
    path_activation = define_path(
        problem, time_layer.magnitude, t_0=-(setup_parameters["num_layers"].magnitude - 1) * time_layer.magnitude
    )
    problem.set_initial_path(path_activation)
    #problem.set_initial_path(0.0)

    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))
    problem.add_sensor(DisplacementSensor([problem.p["layer_length"] / 2, 0, problem.p["layer_height"]]))

    total_time = setup_parameters["num_layers"] * time_layer
    while problem.time <= total_time.to_base_units().magnitude:
        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].min())

    # check residual force bottom
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
    dead_load = (
        problem.p["g"]
        * problem.p["rho"]
        * problem.p["layer_length"]
        * problem.p["num_layers"]
        * problem.p["layer_height"]
        * problem.p["layer_width"]
    )

    print("check", force_bottom_y[-1]/dead_load)
    assert np.isclose(abs(force_bottom_y[-1]/dead_load), 1.0, atol=1e-1)
    

    # check step loading
    disp = np.array(problem.sensors["DisplacementSensor"].data)[:, -1]
    disp_first_layer = disp[0:time_steps_layers]
    disp_rel = disp_first_layer/disp_first_layer[-1]
    print('disp_rel', disp_rel)
    d_steps = len(np.where(disp_rel < 0.9)[0][:])
    print('d_steps', d_steps, 'factor', factor)
    assert d_steps == factor - 1

    # check E modulus evolution over structure (each layer different E)
    if mat.lower() == "linear_elastic":
        if problem.p["time_fct"] == "linear":
            E_bottom_layer = ConcreteAMFC.param_time_fkt(
                problem.time, {"P0": problem.p["E"], "A_P": problem.p["A_E"]}, "linear"
            )
            time_upper = problem.time - (problem.p["num_layers"] - 1) * time_layer.magnitude
            E_upper_layer = ConcreteAMFC.param_time_fkt(
                time_upper, {"P0": problem.p["E"], "A_P": problem.p["A_E"]}, "linear"
            )
            print("E_bottom, E_upper", E_bottom_layer, E_upper_layer)
            print("check", problem.modulus.x.array[:].min() * problem.p["E"], problem.modulus.x.array[:].max()* problem.p["E"])
            assert problem.modulus.x.array[:].min() * problem.p["E"] == pytest.approx(E_upper_layer)
            assert problem.modulus.x.array[:].max() * problem.p["E"] == pytest.approx(E_bottom_layer)
    elif mat.lower() == "mises":
        if problem.p["time_fct"] == "linear":
            p_ka_bottom_layer = ConcreteAMFC.param_time_fkt(
                problem.time, {"P0": problem.p["p_ka"], "A_P": problem.p["A_p_ka"]}, "linear"
            )
            time_upper = problem.time - (problem.p["num_layers"] - 1) * time_layer.magnitude
            p_ka_upper_layer = ConcreteAMFC.param_time_fkt(
                time_upper, {"P0": problem.p["p_ka"], "A_P": problem.p["A_p_ka"]}, "linear"
            )
            print("p_ka_bottom, p_ka_upper", p_ka_bottom_layer, p_ka_upper_layer)
            print(problem.modulus.x.array[:].min(), problem.modulus.x.array[:].max())
            assert problem.modulus.x.array[:].min() == pytest.approx(p_ka_upper_layer)
            assert problem.modulus.x.array[:].max() == pytest.approx(p_ka_bottom_layer)
        #
    if plot:
        # example plotting
        disp = np.array(problem.sensors["DisplacementSensor"].data)[:, -1]
        time = []
        [time.append(ti) for ti in problem.sensors["DisplacementSensor"].time]

        import matplotlib.pylab as plt

        plt.figure(1)
        plt.plot([0] + time, [0] + list(disp), "*-r")
        plt.xlabel("process time")
        plt.ylabel("displacement")
        plt.show()


def define_path(prob, t_diff, t_0=0):
    """create path as layer wise at quadrature space

    one layer by time

    prob: problem
    param: parameter dictionary
    t_diff: time difference between each layer
    t_0: start time for all (0 if static computation)
                            (-end_time last layer if dynamic computation)
    """

    # init path time array
    q_path = prob.rule.create_quadrature_array(prob.mesh, shape=1)

    # get quadrature coordinates with work around since tabulate_dof_coordinates()[:] not possible for quadrature spaces!
    V = df.fem.functionspace(prob.mesh, ("CG", prob.p["degree"],(prob.mesh.topology.dim,)))
    v_cg = df.fem.Function(V)
    if prob.p["dim"] == 2:
        v_cg.interpolate(lambda x: (x[0], x[1]))
        v_cg.x.scatter_forward()
    elif prob.p["dim"] == 3:
        v_cg.interpolate(lambda x: (x[0], x[1], x[2]))
        v_cg.x.scatter_forward()
    positions = QuadratureEvaluator(v_cg, prob.mesh, prob.rule)
    x = positions.evaluate()
    dof_map = np.reshape(x.flatten(), [len(q_path), prob.p["dim"]])

    # select layers
    if prob.p["dim"] == 2:
        # only by layer height - y
        h_CO = np.array(dof_map)[:, 1]
    elif prob.p["dim"] == 3:
        # only by layer height - z
        h_CO = np.array(dof_map)[:, 2]
    h_min = np.arange(0, prob.p["num_layers"] * prob.p["layer_height"], prob.p["layer_height"])
    h_max = np.arange(
        prob.p["layer_height"],
        (prob.p["num_layers"] + 1) * prob.p["layer_height"],
        prob.p["layer_height"],
    )
    #print("h_CO", h_CO)
    #print("h_min", h_min)
    #print("h_max", h_max)
    new_path = np.zeros_like(q_path)
    EPS = 1e-8
    for i in range(0, len(h_min)):
        layer_index = np.where((h_CO > h_min[i] - EPS) & (h_CO <= h_max[i] + EPS))
        new_path[layer_index] = t_0 + (prob.p["num_layers"] - 1 - i) * t_diff

    q_path = new_path

    return q_path


if __name__ == "__main__":
    
    test_am_multiple_layer("linear_elastic", 1, plot=True)
    test_am_multiple_layer("linear_elastic", 2, plot=True)

    #test_am_multiple_layer("mises", 1, plot=True)
    #test_am_multiple_layer("mises", 2, plot=True)
