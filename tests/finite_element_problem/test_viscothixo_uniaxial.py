import os
from pathlib import Path

import numpy as np
import pint
import pytest

from fenicsxconcrete.experimental_setup import SimpleCube
from fenicsxconcrete.finite_element_problem import ConcreteAM, ConcreteViscoDevThixElasticModel
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


def disp_over_time(current_time: pint.Quantity, switch_time: pint.Quantity) -> pint.Quantity:
    """linear ramp of displacement bc over time

    Args:
        t: current time

    Returns: displacement value for given time

    """
    if current_time <= switch_time:
        current_disp = 0.002 * ureg("m") / (switch_time) * current_time
    else:
        current_disp = 0.002 * ureg("m")

    return current_disp


def get_parameters(cur_t, paramsp):
    # compute E_0, E_1, tau for current time and given parameter dic from problem (without units)

    E_0 = ConcreteAM.E_fkt(
        1,
        cur_t,
        {
            "P0": paramsp["E_0"],
            "R_P": paramsp["R_E"],
            "A_P": paramsp["A_E"],
            "tf_P": paramsp["tf_E"],
            "age_0": paramsp["age_0"],
        },
    )

    E_1 = ConcreteAM.E_fkt(
        1,
        cur_t,
        {
            "P0": paramsp["E1_0"],
            "R_P": paramsp["R_E1"],
            "A_P": paramsp["A_E1"],
            "tf_P": paramsp["tf_E1"],
            "age_0": paramsp["age_0"],
        },
    )

    eta = ConcreteAM.E_fkt(
        1,
        cur_t,
        {
            "P0": paramsp["eta_0"],
            "R_P": paramsp["R_eta"],
            "A_P": paramsp["A_eta"],
            "tf_P": paramsp["tf_eta"],
            "age_0": paramsp["age_0"],
        },
    )

    return E_0, E_1, eta / E_1


def material_parameters(parameters, mtype=""):

    if mtype.lower() == "pure_visco":

        _, default_params = ConcreteAM.default_parameters(ConcreteViscoDevThixElasticModel)

        default_params["nu"] = 0.0 * ureg("")  # to compare with 1D analytical solution

    elif mtype.lower() == "visco_thixo":

        _, default_params = ConcreteAM.default_parameters(ConcreteViscoDevThixElasticModel)
        default_params["A_E"] = 70.0e1 * ureg("Pa/s")
        default_params["A_E1"] = 20.0e1 * ureg("Pa/s")
        default_params["A_eta"] = 2.0e1 * ureg("Pa*s/s")

        # much bigger than simulation time to see thix effect!
        default_params["age_0"] = 200.0 * ureg("s")

        default_params["nu"] = 0.0 * ureg("")

    else:
        raise ValueError("material type not implemented")

    return {**parameters, **default_params}


def setup_test_2D(parameters, mech_prob_string, mtype):

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_viscothixo_uniaxial_{parameters['dim']}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete files if they exist (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining experiment parameters
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")

    if parameters["dim"] == 2:
        parameters["stress_state"] = "plane_stress" * ureg("")

    # time
    parameters["dt"] = 0.01 * ureg("s")  # (should be < tau=eta/E_1)
    parameters["load_time"] = 0  # not relevant here

    # material
    parameters = material_parameters(parameters, mtype=mtype)

    # experiment
    experiment = SimpleCube(parameters)

    problem = ConcreteAM(
        experiment, parameters, nonlinear_problem=mech_prob_string, pv_name=file_name, pv_path=data_path
    )

    # add sensors
    if parameters["dim"] == 2:
        problem.add_sensor(StressSensor([0.5, 0.5, 0.0]))
        problem.add_sensor(StrainSensor([0.5, 0.5, 0.0]))
    elif parameters["dim"] == 3:
        problem.add_sensor(StressSensor([0.5, 0.5, 0.5]))
        problem.add_sensor(StrainSensor([0.5, 0.5, 0.5]))

    return problem


@pytest.mark.parametrize("visco_case", ["Cmaxwell", "Ckelvin"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_relaxation(visco_case, dim, mtype, plot=False):
    """
    uniaxial tension test displacement control to check relaxation of visco-thix material class
    """

    parameters = {}
    parameters["dim"] = dim * ureg("")
    parameters["visco_case"] = visco_case * ureg("")
    parameters["density"] = 0.0 * ureg("kg/m**3")
    parameters["strain_state"] = "uniaxial" * ureg("")
    displacement = disp_over_time

    problem = setup_test_2D(parameters, ConcreteViscoDevThixElasticModel, mtype)
    problem.time = -problem.p["dt"]  # to get explizit solution for t=0
    E_o_time = []
    disp_o_time = [0.0]
    total_time = 1.5 * ureg("s")
    while problem.time <= total_time.to_base_units().magnitude:
        # apply increment displacements!!! for time step
        disp_o_time.append(
            displacement((problem.time + 1) * problem.parameters["dt"], problem.parameters["dt"]).to_base_units()
        )
        delta_disp = disp_o_time[-1] - disp_o_time[-2]
        print(delta_disp)
        problem.experiment.apply_displ_load(delta_disp)

        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())

        # store Young's modulus over time
        E_o_time.append(problem.youngsmodulus.vector.array[:].max())

    # get stress over time (Tensor format)
    time = np.array(problem.sensors["StrainSensor"].time)
    if dim == 2:
        # sig_yy and eps_yy in case dim=2
        sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
        eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
    elif dim == 3:
        # sig_zz and eps_zz in case dim=3
        sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
        eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
    #
    print("----relaxation check----")
    print(sig_o_time)
    print(eps_o_time)
    print(time)

    # relaxation check - first and last value
    eps_r = (disp_o_time[-1] / (1.0 * ureg("m"))).magnitude  # (prescriped strain)
    print("prescribed strain", eps_r)

    # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
    # assuming age development much smaller than relaxation time
    E_0, E_1, tau = get_parameters(0, problem.p)
    if problem.p["visco_case"].lower() == "cmaxwell":
        sig0 = E_0 * eps_r + E_1 * eps_r
        sig1 = E_0 * eps_r + E_1 * eps_r * np.exp(-problem.p["dt"] / tau)
        sigend = E_0 * eps_r
    elif problem.p["visco_case"].lower() == "ckelvin":
        sig0 = E_0 * eps_r
        sig1 = E_0 * eps_r / (E_1 + E_0) * (E_1 + E_0 * np.exp(-problem.p["dt"] / tau * (E_0 + E_1) / E_1))
        sigend = (E_0 * E_1) / (E_0 + E_1) * eps_r
    else:
        raise ValueError("visco case not defined")

    print("theory", sig0, sig1, sigend)
    print("computed", sig_o_time[0], sig_o_time[1], sig_o_time[-1])
    # assert (sig_o_time[0] - sig0) / sig0 < 1e-8
    # assert (sig_o_time[-1] - sigend) / sigend < 1e-4

    # get stresses and strain tensors at the end
    # print("stresses", prob.sensors[sensor01.name].data[-1])
    # print("strains", prob.sensors["StrainSensor"].data[-1])
    # check uniaxiality of strain tensor
    if problem.p["dim"] == 2:
        strain_xx = problem.sensors["StrainSensor"].data[-1][0]
        strain_yy = problem.sensors["StrainSensor"].data[-1][-1]
        assert strain_yy == pytest.approx(eps_r)
        assert strain_xx == pytest.approx(-problem.p["nu"] * eps_r)
    elif problem.p["dim"] == 3:
        strain_xx = problem.sensors["StrainSensor"].data[-1][0]
        strain_yy = problem.sensors["StrainSensor"].data[-1][4]
        strain_zz = problem.sensors["StrainSensor"].data[-1][-1]
        assert strain_zz == pytest.approx(eps_r)
        assert strain_xx == pytest.approx(-problem.p["nu"] * eps_r)
        assert strain_yy == pytest.approx(-problem.p["nu"] * eps_r)

    # full analytic 1D solution (for relaxation test -> fits if nu=0 and small enough time steps)
    sig_yy = []
    if problem.p["visco_case"].lower() == "cmaxwell":
        for i in time:
            sig_yy.append(E_0 * eps_r + E_1 * eps_r * np.exp(-i / tau))
    elif problem.p["visco_case"].lower() == "ckelvin":
        for i in time:
            sig_yy.append(E_0 * eps_r / (E_1 + E_0) * (E_1 + E_0 * np.exp(-i / tau * (E_0 + E_1) / E_1)))

    # print("analytic 1D == 2D with nu=0", sig_yy)
    # print("stress over time", sig_o_time)

    ##### plotting #######
    if plot:

        import matplotlib.pyplot as plt

        plt.plot(time, sig_yy, "*r", label="analytic")
        plt.plot(time, sig_o_time, "og", label="FEM")
        plt.xlabel("time [s]")
        plt.ylabel("stress yy [Pa]")
        plt.legend()
        plt.show()


@pytest.mark.parametrize("visco_case", ["Cmaxwell", "Ckelvin"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_creep(visco_case, dim, mtype, plot=False):
    """
    uniaxial tension test with density load as stress control to check creep of visco(-thix) material class
    """

    parameters = {}
    parameters["dim"] = dim
    parameters["visco_case"] = visco_case * ureg("")
    parameters["density"] = 207.0 * ureg("kg/m^3")  # load controlled
    parameters["strain_state"] = "uniaxial" * ureg("")


#     # sensor
#     sensor01 = fenics_concrete.sensors.StressSensor(df.Point(0.5, 0.0))
#     sensor02 = fenics_concrete.sensors.StrainSensor(df.Point(0.5, 0.0))
#
#     prob = setup_test_2D(parameters, mech_prob_string, [sensor01, sensor02], mtype)
#
#     time = []
#     # initialize time and solve!
#     t = 0
#     while t <= prob.p.time:  # time
#         time.append(t)
#         # solve
#         prob.solve(t=t)  # solving this
#         prob.pv_plot(t=t)
#         # prepare next timestep
#         t += prob.p.dt
#
#     # get stress over time
#     if prob.p.dim == 2:
#         # sig_yy and eps_yy in case dim=2
#         sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
#         eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]
#     elif prob.p.dim == 3:
#         # sig_zz and eps_zz in case dim=3
#         sig_o_time = np.array(prob.sensors[sensor01.name].data)[:, -1]
#         eps_o_time = np.array(prob.sensors[sensor02.name].data)[:, -1]
#
#     # relaxation check - first and last value
#     sig_c = sig_o_time[0]
#     assert sig_c == pytest.approx(-prob.p.density * prob.p.g)
#     #
#     # print(prob.p.visco_case)
#     # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
#     # for tests, it is assumed that the total test time is short (age effects neglectable!) but initial concrete age can change parameter!
#     E_0, E_1, tau = get_parameters(0, parameters)
#     if prob.p.visco_case.lower() == "cmaxwell":
#         eps0 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1))
#         epsend = sig_c / E_0
#     elif prob.p.visco_case.lower() == "ckelvin":
#         eps0 = sig_c / E_0
#         epsend = sig_c / E_0 + sig_c / E_1
#     else:
#         raise ValueError("visco case not defined")
#
#     print("theory", eps0, epsend)
#     print("computed", eps_o_time[0], eps_o_time[-1])
#     assert (eps_o_time[0] - eps0) / eps0 < 1e-8
#     assert (eps_o_time[-1] - epsend) / epsend < 1e-4
#
#     # analytic 1D solution (for creep test -> fits if nu=0 and small enough time steps)
#     eps_yy = []
#     if prob.p.visco_case.lower() == "cmaxwell":
#         for i in time:
#             eps_yy.append(sig_c / E_0 * (1 - E_1 / (E_0 + E_1) * np.exp(-i / tau * E_0 / (E_0 + E_1))))
#     elif prob.p.visco_case.lower() == "ckelvin":
#         for i in time:
#             eps_yy.append(sig_c / E_0 + sig_c / E_1 * (1 - np.exp(-i / tau)))
#
#     # print("analytic 1D == 2D with nu=0", eps_yy)
#     # print("stress over time", eps_o_time)
#
#     ##### plotting #######
#     if plot:
#         import matplotlib.pyplot as plt
#
#         plt.plot(time, eps_yy, "*r", label="analytic")
#         plt.plot(time, eps_o_time, "og", label="FEM")
#         plt.xlabel("time [s]")
#         plt.ylabel("eps yy [-]")
#         plt.legend()
#         plt.show()


if __name__ == "__main__":

    test_relaxation("ckelvin", 2, "pure_visco", plot=True)

#     test_creep("ckelvin", 2, "pure_visco", plot=True)
