import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from fenics_constitutive.models import LinearElasticityModel, SpringKelvinModel, SpringMaxwellModel

from fenicsxconcrete.experimental_setup.simple_cube import SimpleCube
from fenicsxconcrete.finite_element_problem.fenics_constitutive import FenicsConstitutive
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("mat", ["linear_elastic", "visco_Kelvin", "visco_Maxwell"])
def test_fc(
    dim: int, mat: Literal["linear_elastic", "visco_Kelvin", "visco_Maxwell"]
) -> None:
    """easy cube test for checking interface fenicsxconcrete - fencis_constitutive
    uniaxial tension test displacement controlled plus body force in case of linear elasticity  """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_mat_fc_uniaxial_{mat}_{dim}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # experiment:
    parameters = {}

    parameters["dim"] = dim * ureg("")
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")
    parameters["strain_state"] = "uniaxial" * ureg("")

    experiment = SimpleCube(parameters)
   
    # material and material parameters:
    if mat == "linear_elastic":
        material_law = LinearElasticityModel
        parameters["E"] = 42000 * ureg("Pa")  # young's modulus
        parameters["nu"] = 0.3 * ureg("")  # poisson ratio

        # apply body force in addition to displacement load
        #experiment.apply_body_force()

    elif mat == "visco_Kelvin" or mat == "visco_Maxwell":
        if mat == "visco_Kelvin":
            material_law = SpringKelvinModel
        elif mat == "visco_Maxwell":
            material_law = SpringMaxwellModel
        parameters["E0"] = 42 * ureg("Pa")
        parameters["E1"] = 10 * ureg("Pa")
        parameters["tau"] = 2 * ureg("s")
        parameters["nu"] = 0.2 * ureg("")  # poisson ratio
    else:
        raise ValueError("material not supported")

    # problem:
    parameters["rho"] = 2000 * ureg("kg/m^3")
    parameters["strain_state"] = "uniaxial" * ureg("")
    parameters["dt"] = 2 * ureg("s")
    parameters["q_degree"] = 4 * ureg("")

    problem = FenicsConstitutive(experiment, parameters, material_law, pv_name=file_name, pv_path=data_path)

    # sensors:
    sensor_location_mid = [0.5, 0.5, 0.5]
    problem.add_sensor(StressSensor(sensor_location_mid))
    problem.add_sensor(DisplacementSensor(where=sensor_location_mid, name="DisplacementSensorMid"))
    problem.add_sensor(DisplacementSensor(where=[0.5, 0.5, 1.0], name="DisplacementSensorTop"))
    problem.add_sensor(ReactionForceSensor())

    # apply displacement load and solve
    displacement = 0.005 * ureg("m")
    if mat == "visco_Kelvin" or mat == "visco_Maxwell":
        total_time = 500  # long enough to see the relaxation of the visco materials
    else: 
        total_time = 10

    print("case:", mat)
    # zero time step for exclude body force displacement for linear elasticity
    if mat == "linear_elastic":
        problem.solve()
        body_force_disp_Mid_z = np.array(problem.sensors["DisplacementSensorMid"].data)[:, 2]
        body_force_disp_Top_z = np.array(problem.sensors["DisplacementSensorTop"].data)[:, 2]
        print("body force disp Mid", body_force_disp_Mid_z)
        print("body force disp Top", body_force_disp_Top_z)
    else:
        body_force_disp_Mid_z = 0. 
        body_force_disp_Top_z = 0.

    while problem.time <= total_time:
        problem.experiment.apply_displ_load(problem.time * displacement/total_time)
        problem.solve()
        problem.pv_plot()
     
    #print("disp Top", np.array(problem.sensors["DisplacementSensorTop"].data)[:, 2])
    #print("disp Middle", np.array(problem.sensors["DisplacementSensorMid"].data)[:, 2])
    # print("Stress Sensor Data:", np.array(problem.sensors["StressSensor"].data))
    #print("reaction force:", np.array(problem.sensors["ReactionForceSensor"].data)) 
 
    # check max displacement at top
    assert np.isclose(
        np.array(problem.sensors["DisplacementSensorTop"].data)[-1, 2] - body_force_disp_Top_z,
        abs(displacement.magnitude),
        rtol=1e-2,
    )
    # displacement in middle 1/2 of total per time step
    delta_disp_z = np.array(problem.sensors["DisplacementSensorMid"].data)[:, 2] - body_force_disp_Mid_z
    assert np.isclose(delta_disp_z[-1], abs(displacement.magnitude) / 2, rtol=1e-2)
    # check reaction force (relaxation test for visco materials from analytical solution)
    if mat == "linear_elastic":
        stress_final_ana = parameters["E"].magnitude * displacement.magnitude / 1.0
    elif mat == "visco_Kelvin":
        stress_final_ana = (
            parameters["E0"].magnitude * parameters["E1"].magnitude / (parameters["E0"].magnitude + parameters["E1"].magnitude) 
            * displacement.magnitude / 1.0
        )
    elif mat == "visco_Maxwell":
        stress_final_ana = parameters["E0"].magnitude * displacement.magnitude / 1.0
    
    print("reaction force:", np.array(problem.sensors["ReactionForceSensor"].data)[:,2][-1], stress_final_ana)
    print("stress sensor:", np.array(problem.sensors["StressSensor"].data)[:, 2][-1])
    assert np.isclose(
        np.array(problem.sensors["ReactionForceSensor"].data)[:,2][-1],
        stress_final_ana, 
        rtol=1e-2,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test_fc(3, "linear_elastic")
    test_fc(3, "visco_Kelvin")
    test_fc(3, "visco_Maxwell")
