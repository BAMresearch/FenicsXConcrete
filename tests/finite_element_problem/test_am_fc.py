import os
from pathlib import Path

import numpy as np
import pytest

# for know copy material law from fenics-constitutive to tests/finite_element_problem should be a module later
from linear_elasticity_model import LinearElasticityModel
from mises_plasticity_isotropic_hardening import VonMises3D

from fenicsxconcrete.experimental_setup import AmMultipleLayers, SimpleCube
from fenicsxconcrete.finite_element_problem.concrete_am_fc import ConcreteAMFC
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, ureg

#
# @pytest.mark.parametrize("dim", [3])
# @pytest.mark.parametrize("mat", ["linear_elastic"])
# @pytest.mark.parametrize("bc", ["disp", "force"])
# def test_fc(dim: int, mat: str, bc: str) -> None:
#     """easy cube test for checking concrete_am_fc material problem class
#     uniaxial tension test"""
#
#     # setup paths and directories
#     data_dir = "data_files"
#     data_path = Path(__file__).parent / data_dir
#
#     # define file name and path for paraview output
#     file_name = f"test_am_fc_uniaxial_{dim}d"
#     files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
#     # delete file if it exists (only relevant for local tests)
#     for file in files:
#         if file.is_file():
#             os.remove(file)
#
#     # defining experiment parameters
#     parameters = {}
#
#     parameters["dim"] = dim * ureg("")
#     parameters["num_elements_length"] = 2 * ureg("")
#     parameters["num_elements_height"] = 2 * ureg("")
#     parameters["num_elements_width"] = 2 * ureg("")
#     parameters["q_degree"] = 4 * ureg("")
#
#     experiment = SimpleCube(parameters)
#     if bc == "force":
#         experiment.apply_body_force()
#
#     # material:
#     if mat == "linear_elastic":
#         material_law = LinearElasticityModel
#         parameters["E"] = 42000 * ureg("Pa")  # young's modulus
#         parameters["nu"] = 0.3 * ureg("")  # poisson ratio
#         parameters["A_E"] = 4000 * ureg("Pa/s")  # young's modulus rate over time
#         parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
#     elif mat == "mises":
#         material_law = VonMises3D
#         parameters["p_ka"] = 175000 * ureg("MPa")  # bulk modulus
#         parameters["p_mu"] = 80769 * ureg("MPa")  # shear modulus
#         parameters["p_y0"] = 1200 * ureg("MPa")  # initial yield stress
#         parameters["p_y00"] = 2500 * ureg("MPa")  # final yield stress
#         parameters["p_w"] = 200 * ureg("")  # saturation parameter
#     else:
#         raise ValueError("material not supported")
#
#     # problem:
#     parameters["rho"] = 2000 * ureg("kg/m^3")
#     parameters["strain_state"] = "uniaxial" * ureg("")
#     parameters["dt"] = 0.1 * ureg("s")
#     parameters["q_degree"] = 4 * ureg("")
#
#     problem = ConcreteAMFC(experiment, parameters, material_law, pv_name=file_name, pv_path=data_path)
#
#     # sensors:
#     sensor_location = [0.5, 0.5, 0.5]
#     problem.add_sensor(StressSensor(sensor_location))
#     problem.add_sensor(DisplacementSensor(sensor_location))
#     problem.add_sensor(ReactionForceSensor())
#
#     # apply displacement load and solve
#     displacement = 0.005 * ureg("m")
#     total_time = 1.0
#     while problem.time <= total_time:
#         problem.experiment.apply_displ_load(problem.time * displacement)
#         problem.solve()
#         problem.pv_plot()
#         print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())
#
#     disp_result = problem.sensors["DisplacementSensor"].get_last_entry().magnitude
#     force_result = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
#
#     stress_result = problem.sensors["StressSensor"].get_last_entry().magnitude
#     print("results", disp_result, force_result, stress_result)
#
#     # check
#     assert np.isclose(disp_result[0], disp_result[1], rtol=1e-2)  # uniaxial tension
#
#     if bc == "disp":
#         # max displacement should be equal to the applied displacement
#         assert np.isclose(abs(problem.fields.displacement.x.array[:]).max(), abs(displacement.magnitude), rtol=1e-2)
#     elif bc == "force":
#         assert abs(problem.fields.displacement.x.array[:]).max() > abs(displacement.magnitude)
#
#     # check time dependent properties
#     if mat == "linear_elastic":
#         assert np.isclose(
#             np.diff(np.diff(force_result)).mean(),
#             problem.p["A_E"] * problem.p["dt"] * problem.p["dt"] * displacement.magnitude / problem.p["height"],
#             rtol=1e-2,
#         )


def set_test_parameters(mat: str = "linear_elastic") -> Parameters:
    """set up a test parameter set

    Args:
        dim: dimension of problem

    Returns: filled instance of Parameters

    """
    setup_parameters = {}

    setup_parameters["dim"] = 3 * ureg("")
    # setup_parameters["stress_state"] = "plane_strain"
    setup_parameters["num_layers"] = 5 * ureg("")  # changed in single layer test!!
    setup_parameters["layer_height"] = 1 / 100 * ureg("m")  # y (2D), z (3D)
    setup_parameters["layer_length"] = 50 / 100 * ureg("m")  # x
    setup_parameters["layer_width"] = 5 / 100 * ureg("m")  # y (3D)

    setup_parameters["num_elements_layer_length"] = 10 * ureg("")
    setup_parameters["num_elements_layer_height"] = 1 * ureg("")
    setup_parameters["num_elements_layer_width"] = 2 * ureg("")

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
        setup_parameters["p_ka"] = 175000 * ureg("MPa")  # bulk modulus
        setup_parameters["p_mu"] = 80769 * ureg("MPa")  # shear modulus
        setup_parameters["p_y0"] = 1200 * ureg("MPa")  # initial yield stress
        setup_parameters["p_y00"] = 2500 * ureg("MPa")  # final yield stress
        setup_parameters["p_w"] = 200 * ureg("")  # saturation parameter
    else:
        raise ValueError("material not supported")

    return setup_parameters, material_law


@pytest.mark.parametrize("mat", ["linear_elastic"])
@pytest.mark.parametrize("factor", [1, 2])
def test_am_single_layer(mat: str, factor: int) -> None:
    """single layer test

    one layer build immediately and lying for a given time

    Args:
        dimension: dimension
        factor: length of load_time = factor * dt
    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_am_fc_single_layer"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters, material_law = set_test_parameters()
    setup_parameters["num_layers"] = 1 * ureg("")

    # solving parameters
    solve_parameters = {}
    solve_parameters["time"] = 6 * 60 * ureg("s")

    # defining different loading
    setup_parameters["dt"] = 60 * ureg("s")
    # setup_parameters["load_time"] = factor * setup_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem

    experiment = AmMultipleLayers(setup_parameters)

    problem = ConcreteAMFC(experiment, setup_parameters, material_law, pv_name=file_name, pv_path=data_path)
    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))

    E_o_time = []
    total_time = 6 * 60 * ureg("s")
    while problem.time <= total_time.to_base_units().magnitude:
        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())
        E_o_time.append(problem.modulus.vector.array[:].max())

    # check reaction force
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
    dead_load = (
        problem.p["g"]
        * problem.p["rho"]
        * problem.p["layer_length"]
        * problem.p["num_layers"]
        * problem.p["layer_height"]
        * problem.p["layer_width"]
    )

    # # dead load of full structure
    print("Check", force_bottom_y, dead_load)
    assert force_bottom_y.mean() == pytest.approx(-dead_load)

    # check stresses change according to Emodul change
    sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, 2]  # zz

    if factor == 1:
        # instance loading -> no changes
        assert sum(np.diff(sig_o_time)) == pytest.approx(0, abs=1e-8)
    # elif factor == 2:
    #     # ratio sig/eps t=0 to sig/eps t=0+dt
    #     E_ratio_computed = (sig_o_time[0] / eps_o_time[0]) / (np.diff(sig_o_time)[0] / np.diff(eps_o_time)[0])
    #     assert E_ratio_computed == pytest.approx(E_o_time[0] / E_o_time[1])
    #     # after second time step nothing should change anymore
    #     assert sum(np.diff(sig_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)
    #     assert sum(np.diff(eps_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)

    if mat == "linear_elastic":
        if problem.p["time_fct"] == "linear":
            assert np.isclose(np.diff(E_o_time).mean(), problem.p["A_E"] * problem.p["dt"] / problem.p["E"], rtol=1e-2)


if __name__ == "__main__":

    # test_fc(3, "linear_elastic", "disp")

    # test_fc(3, "linear_elastic", "force")

    test_am_single_layer("linear_elastic", 1)
