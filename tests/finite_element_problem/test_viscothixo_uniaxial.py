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


def material_parameters(mtype=""):

    if mtype.lower() == "pure_visco":

        _, default_params = ConcreteAM.default_parameters(ConcreteViscoDevThixElasticModel)

        default_params["nu"] = 0.0 * ureg("")  # to compare with 1D analytical solution

    elif mtype.lower() == "visco_thixo":

        _, default_params = ConcreteAM.default_parameters(ConcreteViscoDevThixElasticModel)
        default_params["R_E"] = 70.0e1 * ureg("Pa/s")
        default_params["R_E1"] = 20.0e1 * ureg("Pa/s")
        default_params["R_eta"] = 2.0e1 * ureg("Pa*s/s")
        default_params["tf_E"] = 20.0 * ureg("s")  # < age
        default_params["tf_E1"] = 20.0 * ureg("s")
        default_params["tf_eta"] = 20.0 * ureg("s")

        # much bigger than simulation time to see thix effect!
        default_params["age_0"] = 200.0 * ureg("s")

        default_params["nu"] = 0.0 * ureg("")

    else:
        raise ValueError("material type not implemented")

    return default_params


def setup_test_2D(parameters, mtype):

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
    parameters["dt"] = 0.01 * ureg("s")  # 0.001 (should be < tau=eta/E_1)
    parameters["load_time"] = parameters["dt"]  # only relevant for creep

    # experiment
    experiment = SimpleCube(parameters)

    problem = ConcreteAM(
        experiment,
        parameters,
        nonlinear_problem=ConcreteViscoDevThixElasticModel,
        pv_name=file_name,
        pv_path=data_path,
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
@pytest.mark.parametrize("dim", [2])  # [2,3]
@pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
def test_relaxation(visco_case, dim, mtype, plot=False):
    """
    uniaxial tension test displacement control to check relaxation of visco-thix material class
    """
    # get default material parameters
    parameters = material_parameters(mtype=mtype)
    # change parameters for individual test
    parameters["dim"] = dim * ureg("")
    parameters["visco_case"] = visco_case * ureg("")
    parameters["density"] = 0.0 * ureg("kg/m**3")
    parameters["strain_state"] = "uniaxial" * ureg("")
    disp = 0.002 * ureg("m")

    problem = setup_test_2D(parameters, mtype)

    # apply load at time zero using a very small time step
    dt_p = problem.p["dt"]
    problem.p["dt"] = 1e-10
    problem.experiment.apply_displ_load(disp.to_base_units())
    problem.solve()
    problem.pv_plot()
    problem.p["dt"] = dt_p

    # further time step normal way
    total_time = 2.0 * ureg("s")
    while problem.time <= total_time.to_base_units().magnitude:
        # no further loading
        problem.experiment.apply_displ_load(0.0 * ureg("m"))
        print("solve for time", problem.time + problem.p["dt"])
        problem.solve()
        problem.pv_plot()

        print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())

        print("strain", problem.time, problem.sensors["StrainSensor"].data[-1])
        print("stress", problem.time, problem.sensors["StressSensor"].data[-1])
        print("visco strain?", problem.time, problem.q_fields.visco_strain.x.array[:].max())

    # get stress over time (Tensor format)
    time = np.array(problem.sensors["StrainSensor"].time)
    # stress and strain in case of dim=2 yy in case of dim=3 zz
    sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
    eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
    print("time", time)
    print("sig_o_time", sig_o_time)
    print("eps_o_time", eps_o_time)

    #
    problem.mechanics_problem.evaluate_material()
    print("""""")
    # print("----relaxation check----")
    # relaxation check - first and last value
    eps_r = (disp / (1.0 * ureg("m"))).magnitude  # (prescriped strain)
    # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
    # assuming age > tf and structuration rate A_i == 0
    E_0, E_1, tau = get_parameters(0, problem.p)
    if problem.p["visco_case"].lower() == "cmaxwell":
        # sig0 = E_0 * eps_r + E_1 * eps_r
        sig1 = E_0 * eps_r + E_1 * eps_r * np.exp(-1e-10 / tau)
        sigend = E_0 * eps_r
    elif problem.p["visco_case"].lower() == "ckelvin":
        # sig0 = E_0 * eps_r
        sig1 = E_0 * eps_r / (E_1 + E_0) * (E_1 + E_0 * np.exp(-1e-10 / tau * (E_0 + E_1) / E_1))
        sigend = (E_0 * E_1) / (E_0 + E_1) * eps_r
    else:
        raise ValueError("visco case not defined")

    print("theory", sig1, sigend)
    print("computed", sig_o_time[0], sig_o_time[-1])
    assert sig_o_time[0] == pytest.approx(sig1)
    assert sig_o_time[-1] == pytest.approx(sigend)

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

    ##### plotting #######
    if plot:

        # full analytic 1D solution (for relaxation test -> fits if nu=0 and small enough time steps)
        sig_yy = []
        if problem.p["visco_case"].lower() == "cmaxwell":
            for i in time:
                sig_yy.append(E_0 * eps_r + E_1 * eps_r * np.exp(-i / tau))
        elif problem.p["visco_case"].lower() == "ckelvin":
            for i in time:
                sig_yy.append(E_0 * eps_r / (E_1 + E_0) * (E_1 + E_0 * np.exp(-i / tau * (E_0 + E_1) / E_1)))

        import matplotlib.pyplot as plt

        plt.plot(time, sig_yy, "*r", label="analytic")
        plt.plot(time, sig_o_time, "og", label="FEM")
        plt.xlabel("time [s]")
        plt.ylabel("stress yy [Pa]")
        plt.legend()
        plt.show()


#
# @pytest.mark.parametrize("visco_case", ["Cmaxwell", "Ckelvin"])
# @pytest.mark.parametrize("dim", [2, 3])
# @pytest.mark.parametrize("mtype", ["pure_visco", "visco_thixo"])
# def test_creep(visco_case, dim, mtype, plot=False):
#     """
#     uniaxial tension test with density load as stress control to check creep of visco(-thix) material class
#     """
#
#     # get default material parameters
#     parameters = material_parameters(mtype=mtype)
#     # change parameters for individual test
#     parameters["dim"] = dim * ureg("")
#     parameters["visco_case"] = visco_case * ureg("")
#     parameters["density"] = 0.2070 * ureg("kg/m^3")  # load controlled
#     parameters["strain_state"] = "stress_controlled" * ureg("")
#
#     problem = setup_test_2D(parameters, mtype)
#
#     print("EG", problem.p["density"] * problem.p["g"])
#
#     # apply load at time zero using a very small time step
#     dt_p = problem.p["dt"]
#     problem.p["dt"] = 1e-10
#     problem.p["load_time"] = problem.p["dt"]
#     problem.solve()
#     problem.pv_plot()
#     problem.p["dt"] = dt_p
#     input()
#
#     sig_c = 0.0
#     E_0, E_1, tau = get_parameters(0, problem.p)
#     if problem.p["visco_case"].lower() == "cmaxwell":
#         eps0 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1))
#         eps1 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1) * np.exp(-1e-10 / tau * E_0 / (E_0 + E_1)))
#         epsend = sig_c / E_0
#     elif problem.p["visco_case"].lower() == "ckelvin":
#         eps0 = sig_c / E_0
#         eps1 = sig_c / E_0 + sig_c / E_1 * (1 - np.exp(-1e-10 / tau))
#         epsend = sig_c / E_0 + sig_c / E_1
#     else:
#         raise ValueError("visco case not defined")
#     print("theory", eps0, eps1, epsend)
#
#     ##
#     E_o_time = []
#     total_time = 1.5 * ureg("s")
#     while problem.time <= total_time.to_base_units().magnitude:
#         print(problem.time)
#         input()
#         problem.solve()
#         problem.pv_plot()
#         print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())
#         input()
#         # store Young's modulus over time
#         E_o_time.append(problem.youngsmodulus.vector.array[:].max())
#
#     # get stress over time (Tensor format)
#     time = np.array(problem.sensors["StrainSensor"].time)
#     # stress and strain in case of dim=2 yy in case of dim=3 zz
#     sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
#     eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
#
#     #
#     print("----creep check----")
#     print(sig_o_time)
#     print(eps_o_time)
#     print(time)
#
#     #
#     #     # relaxation check - first and last value
#     #     sig_c = sig_o_time[0]
#     sig_c = problem.p["density"] * problem.p["g"]
#     sig_c = 0.0
#     print("check sig_c", sig_c)
#     #     assert sig_c == pytest.approx(-prob.p.density * prob.p.g)
#     #     #
#     #     # print(prob.p.visco_case)
#     #     # analytic case just for CONSTANT parameters over time (otherwise analytic integration not valid)
#     #     # for tests, it is assumed that the total test time is short (age effects neglectable!) but initial concrete age can change parameter!
#     E_0, E_1, tau = get_parameters(0, problem.p)
#     if problem.p["visco_case"].lower() == "cmaxwell":
#         eps0 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1))
#         eps1 = sig_c / E_0 * (1 - E_1 / (E_0 + E_1) * np.exp(-1e-10 / tau * E_0 / (E_0 + E_1)))
#         epsend = sig_c / E_0
#     elif problem.p["visco_case"].lower() == "ckelvin":
#         eps0 = sig_c / E_0
#         eps1 = sig_c / E_0 + sig_c / E_1 * (1 - np.exp(-1e-10 / tau))
#         epsend = sig_c / E_0 + sig_c / E_1
#     else:
#         raise ValueError("visco case not defined")
#
#     print("theory", eps0, eps1, epsend)
#     print("computed", eps_o_time[0], eps_o_time[-1])
#     #     assert (eps_o_time[0] - eps0) / eps0 < 1e-8
#     #     assert (eps_o_time[-1] - epsend) / epsend < 1e-4
#     #
#     # analytic 1D solution (for creep test -> fits if nu=0 and small enough time steps)
#     eps_yy = []
#     if problem.p["visco_case"].lower() == "cmaxwell":
#         for i in time:
#             eps_yy.append(sig_c / E_0 * (1 - E_1 / (E_0 + E_1) * np.exp(-i / tau * E_0 / (E_0 + E_1))))
#     elif problem.p["visco_case"].lower() == "ckelvin":
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
#
#
if __name__ == "__main__":

    # test_relaxation("ckelvin", 2, "pure_visco", plot=True)
    # test_relaxation("cmaxwell", 2, "pure_visco", plot=True)
    test_relaxation("ckelvin", 2, "pure_visco", plot=True)

    # test_creep("ckelvin", 2, "pure_visco", plot=True)
    # test_creep("cmaxwell", 2, "pure_visco", plot=True)
