import os
from pathlib import Path

import pint
import pytest

from fenicsxconcrete.experimental_setup import SimpleCube
from fenicsxconcrete.finite_element_problem import ConcreteAM, ConcreteThixElasticModel
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
        current_disp = 0.1 * ureg("m") / (switch_time) * current_time
    else:
        current_disp = 0.1 * ureg("m")

    return current_disp


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_disp(dim: int, degree: int):
    """uniaxial test displacement controlled

    Args:
        dim: dimension of the test (2 or 3)
        degree: polynominal degree
    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_thixotropy_uniaxial_{dim}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining experiment parameters
    parameters = {}

    parameters["dim"] = dim * ureg("")
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")

    if dim == 2:
        parameters["stress_state"] = "plane_stress" * ureg("")

    # setting up the problem
    experiment = SimpleCube(parameters)

    # get default parameters and change accordingly to cases
    des = ConcreteAM.parameter_description()
    print(des)

    _, default_params = ConcreteAM.default_parameters(ConcreteThixElasticModel)
    parameters.update(default_params)
    parameters["degree"] = degree * ureg("")
    if dim == 3:
        parameters["q_degree"] = 4 * ureg("")

    # defining time parameters
    parameters["dt"] = 1 * 60 * ureg("s")
    total_time = 6 * 60 * ureg("s")

    # displacement controlled uniaxial test with no body force
    parameters["strain_state"] = "uniaxial" * ureg("")
    displacement = disp_over_time
    parameters["density"] = 0 * ureg("kg/m^3")  # no body force!!

    problem = ConcreteAM(
        experiment, parameters, nonlinear_problem=ConcreteThixElasticModel, pv_name=file_name, pv_path=data_path
    )

    # add sensors
    if dim == 2:
        problem.add_sensor(StressSensor([0.5, 0.5, 0.0]))
        problem.add_sensor(StrainSensor([0.5, 0.5, 0.0]))
    elif dim == 3:
        problem.add_sensor(StressSensor([0.5, 0.5, 0.5]))
        problem.add_sensor(StrainSensor([0.5, 0.5, 0.5]))

    E_o_time = []
    disp_o_time = [0.0]
    while problem.time <= total_time.to_base_units().magnitude:
        # apply increment displacements!!!
        disp_o_time.append(
            displacement((problem.time + problem.p["dt"]) * ureg("s"), 2 * parameters["dt"]).to_base_units()
        )
        delta_disp = disp_o_time[-1] - disp_o_time[-2]
        problem.experiment.apply_displ_load(delta_disp)

        problem.solve()
        problem.pv_plot()
        print("problem time", problem.time)
        print("computed disp", problem.fields.displacement.x.array[:].max())

        # store Young's modulus over time
        E_o_time.append(problem.youngsmodulus.vector.array[:].max())

    print("Stress sensor", problem.sensors["StressSensor"].data)
    print("strain sensor", problem.sensors["StrainSensor"].data)
    print("time", problem.sensors["StrainSensor"].time)
    print("E modul", E_o_time)

    check_disp_case(problem, parameters["dt"], E_o_time)


def check_disp_case(problem: ConcreteAM, dt: pint.Quantity, E_o_time: list[float]) -> None:
    """checks for displacement controlled version

    Args:
        problem: concreteam problem instance
        dt: time step parameter
        E_o_time: Youngs modulus values over time

    """

    disp_at_end = disp_over_time(problem.sensors["StrainSensor"].time[-1] * ureg("s"), 2 * dt).to_base_units()
    analytic_eps = (disp_at_end / (1.0 * ureg("m"))).magnitude
    disp_dt1 = disp_over_time(problem.sensors["StrainSensor"].time[0] * ureg("s"), 2 * dt).to_base_units()
    analytic_eps_dt1 = (disp_dt1 / (1.0 * ureg("m"))).magnitude
    print(analytic_eps, analytic_eps_dt1, disp_at_end, disp_dt1)

    if problem.p["dim"] == 2:
        # standard uniaxial checks for last time step
        # strain in yy direction
        assert problem.sensors["StrainSensor"].data[-1][-1] == pytest.approx(analytic_eps)
        # strain in xx direction
        assert problem.sensors["StrainSensor"].data[-1][0] == pytest.approx(-problem.p["nu"] * analytic_eps)
        # strain in xy and yx direction
        assert problem.sensors["StrainSensor"].data[-1][1] == pytest.approx(
            problem.sensors["StrainSensor"].data[-1][2]
        )
        assert problem.sensors["StrainSensor"].data[-1][1] == pytest.approx(0.0)

        # thix related tests
        # thix tests stress in yy first time step
        assert problem.sensors["StressSensor"].data[0][-1] == pytest.approx((analytic_eps_dt1 * E_o_time[0]))
        # stress delta between last time steps
        assert problem.sensors["StressSensor"].data[-1][-1] - problem.sensors["StressSensor"].data[-2][
            -1
        ] == pytest.approx(0.0)
    elif problem.p["dim"] == 3:
        # standard uniaxial checks for last time step
        # strain in zz direction
        assert problem.sensors["StrainSensor"].data[-1][-1] == pytest.approx(analytic_eps)
        # strain in yy direction
        assert problem.sensors["StrainSensor"].data[-1][4] == pytest.approx(-problem.p["nu"] * analytic_eps)
        # strain in xx direction
        assert problem.sensors["StrainSensor"].data[-1][0] == pytest.approx(-problem.p["nu"] * analytic_eps)
        # shear strains
        sum_mixed_strains = (
            problem.sensors["StrainSensor"].data[-1][1]  # xy
            - problem.sensors["StrainSensor"].data[-1][3]  # yx
            - problem.sensors["StrainSensor"].data[-1][2]  # xz
            - problem.sensors["StrainSensor"].data[-1][6]  # zx
            - problem.sensors["StrainSensor"].data[-1][5]  # yz
            - problem.sensors["StrainSensor"].data[-1][7]  # zy
        )
        assert sum_mixed_strains == pytest.approx(0.0)

        # thix related tests
        # thix tests stress in zz direction first time step
        assert problem.sensors["StressSensor"].data[0][-1] == pytest.approx(analytic_eps_dt1 * E_o_time[0])
        # stress delta between last time steps
        assert problem.sensors["StressSensor"].data[-1][-1] - problem.sensors["StressSensor"].data[-2][
            -1
        ] == pytest.approx(0.0)

    # check changing youngs modulus
    if problem.time < problem.p["tf_E"]:
        E_end = problem.p["E_0"] + problem.p["R_E"] * (problem.time + problem.p["age_0"])
    else:
        E_end = (
            problem.p["E_0"]
            + problem.p["R_E"] * problem.p["tf_E"]
            + problem.p["A_E"] * (problem.time + problem.p["age_0"] - problem.p["tf_E"])
        )
    assert E_o_time[-1] == pytest.approx(E_end)


# if __name__ == "__main__":
#
#     test_disp(2, 2)
#
#     # test_disp(3, 2)
