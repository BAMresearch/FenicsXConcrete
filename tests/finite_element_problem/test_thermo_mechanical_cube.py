import os
from pathlib import Path

import dolfinx as df
import numpy as np
import pytest

from fenicsxconcrete.boundary_conditions import BoundaryConditions
from fenicsxconcrete.experimental_setup import SimpleCube
from fenicsxconcrete.finite_element_problem import ConcreteThermoMechanical, LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.doh_sensor import DOHSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_mechanical_only(dim: int) -> None:
    # defining experiment parameters
    parameters_exp = {}
    parameters_exp["dim"] = dim * ureg("")
    parameters_exp["num_elements_length"] = 2 * ureg("")
    parameters_exp["num_elements_height"] = 2 * ureg("")
    parameters_exp["num_elements_width"] = 2 * ureg("")
    parameters_exp["strain_state"] = "uniaxial" * ureg("")

    displacement = 0.01 * ureg("m")

    parameters = {}
    parameters["rho"] = 7750 * ureg("kg/m^3")
    parameters["E"] = 210e9 * ureg("N/m^2")
    parameters["nu"] = 0.28 * ureg("")

    experiment = SimpleCube(parameters_exp)

    problem_elastic = LinearElasticity(experiment, parameters, pv_name=f"pure_mechanical_test_{dim}", pv_path="")

    _, parameters_thermo = ConcreteThermoMechanical.default_parameters()
    parameters_thermo["nu"] = parameters["nu"].copy()
    parameters_thermo["E_28"] = parameters["E"].copy()
    parameters_thermo["q_degree"] = 4 * ureg("")

    problem_thermo_mechanical = ConcreteThermoMechanical(
        experiment, parameters_thermo, pv_name=f"thermo_mechanical_test_{dim}", pv_path=""
    )

    # apply displacement load and solve
    experiment.apply_displ_load(displacement)
    experiment.apply_body_force()

    problem_elastic.solve()
    problem_elastic.pv_plot()

    # problem_thermo_mechanical.experiment.apply_displ_load(displacement)
    problem_thermo_mechanical.temperature_problem.q_alpha.vector.array[:] = parameters_thermo["alpha_max"].magnitude

    problem_thermo_mechanical.mechanics_solver.solve(problem_thermo_mechanical.fields.displacement)
    problem_thermo_mechanical.pv_plot()

    assert problem_thermo_mechanical.q_fields.youngs_modulus.vector.array[:] == pytest.approx(
        parameters["E"].magnitude
    )

    np.testing.assert_allclose(
        problem_thermo_mechanical.fields.displacement.vector.array,
        problem_elastic.fields.displacement.vector.array,
        rtol=1e-4,
    )


class LegacyMinimalCube(SimpleCube):
    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        bc_generator = BoundaryConditions(self.mesh, V)

        bc_generator.add_dirichlet_bc(
            df.fem.Constant(self.mesh, np.zeros(self.p["dim"])),
            boundary=self.boundary_bottom(),
            method="geometrical",
            entity_dim=self.mesh.topology.dim - 1,
        )

        return bc_generator.bcs


@pytest.mark.parametrize("dim", [2, 3])
def test_hydration_with_body_forces(dim: int):
    # This test relies on data from the old repository

    parameters = {}  # using the current default values
    # general
    # parameters["log_level"] = "WARNING" * ureg("")
    # mesh
    # parameters["mesh_setting"] = "left/right" * ureg("") # default boundary setting
    parameters["dim"] = 3 * ureg("")
    # parameters["mesh_density"] = 2 * ureg("")
    parameters["length"] = 1.0 * ureg("m")
    parameters["width"] = 1.0 * ureg("m")
    parameters["height"] = 1.0 * ureg("m")

    # Differs from the old repository
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")

    # temperature boundary
    # parameters["bc_setting"] = "full" * ureg("")
    parameters["T_0"] = ureg.Quantity(20.0, ureg.degC)  # inital concrete temperature
    parameters["T_bc1"] = ureg.Quantity(20.0, ureg.degC)  # temperature boundary value 1

    # Differs from the old repository
    parameters["rho"] = 2350 * ureg("kg/m^3")  # in kg/m^3 density of concrete

    parameters["density_binder"] = 1440 * ureg("kg/m^3")  # in kg/m^3 density of the binder
    parameters["thermal_cond"] = 2.0 * ureg(
        "W/(m^3*K)"
    )  # effective thermal conductivity, approx in Wm^-3K^-1, concrete!
    # self.specific_heat_capacity = 9000  # effective specific heat capacity in J kg⁻1 K⁻1
    parameters["vol_heat_cap"] = 2.4e6 * ureg("J/(m^3 * K)")  # volumetric heat cap J/(m3 K)
    # parameters["b_ratio"] = 0.2  # volume percentage of binder
    parameters["Q_pot"] = 500e3 * ureg("J/kg")  # potential heat per weight of binder in J/kg
    parameters["Q_inf"] = 144000000 * ureg("J/m^3")
    # p['Q_inf'] = self.Q_pot * self.density_binder * self.b_ratio  # potential heat per concrete volume in J/m3
    parameters["B1"] = 2.916e-4 * ureg("1/s")  # in 1/s
    parameters["B2"] = 0.0024229 * ureg("1/s")  # -
    parameters["eta"] = 5.554 * ureg("")  # something about diffusion
    parameters["alpha_max"] = 0.87 * ureg("")  # also possible to approximate based on equation with w/c
    parameters["alpha_tx"] = 0.68 * ureg("")  # also possible to approximate based on equation with w/c
    parameters["E_act"] = 5653 * 8.3145 * ureg("J*mol^-1")  # activation energy in Jmol^-1
    parameters["T_ref"] = ureg.Quantity(25.0, ureg.degC)  # reference temperature in degree celsius
    # setting for temperature adjustment
    parameters["temp_adjust_law"] = "exponential" * ureg("")
    # polinomial degree
    parameters["degree"] = 2 * ureg("")  # default boundary setting
    parameters["q_degree"] = 4 * ureg("")
    ### paramters for mechanics problem
    parameters["E_28"] = 42000000.0 * ureg("N/m^2")  # Youngs Modulus N/m2 or something...
    parameters["nu"] = 0.2 * ureg("")  # Poissons Ratio
    # required paramters for alpha to E mapping
    parameters["alpha_t"] = 0.2 * ureg("")
    parameters["alpha_0"] = 0.05 * ureg("")
    parameters["a_E"] = 0.6 * ureg("")
    # required paramters for alpha to tensile and compressive stiffness mapping
    parameters["fc_inf"] = 6210000 * ureg("")
    parameters["a_fc"] = 1.2 * ureg("")
    parameters["ft_inf"] = 467000 * ureg("")
    parameters["a_ft"] = 1.0 * ureg("")
    parameters["igc"] = 8.3145 * ureg("J/K/mol")
    parameters["evolution_ft"] = "False" * ureg("")
    parameters["dt"] = 60.0 * ureg("min")

    experiment = LegacyMinimalCube(parameters)
    experiment.apply_body_force()
    problem = ConcreteThermoMechanical(
        experiment=experiment, parameters=parameters, pv_name=f"thermo_mechanical_stuff_{dim}"
    )
    # problem = fenics_concrete.ConcreteThermoMechanical(experiment=experiment, parameters=parameters, vmapoutput=False)

    doh_sensor = DOHSensor([0.25, 0.25, 0.25], name="doh")

    problem.add_sensor(doh_sensor)
    # problem.add_sensor(t_sensor)
    problem.logger.setLevel("INFO")
    # data for time stepping

    # set time step
    # problem.set_timestep()  # for time integration scheme

    # initialize time
    t = problem.p["dt"]  # first time step time
    problem.time = t
    t_list = []
    u_list = []
    temperature_list = []
    doh = 0
    while doh < parameters["alpha_tx"]:  # time
        # for i in range(6):
        # solve temp-hydration-mechanics
        t_list.append(problem.time)
        problem.solve()  # solving this
        u_list.append(problem.fields.displacement.vector.array[:])
        temperature_list.append(problem.fields.temperature.vector.array[:])
        try:
            problem.pv_plot()
        except:
            print("Plotting failed")
        # prepare next timestep
        print(problem.time)
        # get last measure value
        doh = problem.sensors[doh_sensor.name].data[-1]
        print(doh, parameters["alpha_tx"])
    dof_map_u = problem.fields.displacement.function_space.tabulate_dof_coordinates()
    dof_map_t = problem.fields.temperature.function_space.tabulate_dof_coordinates()
    np.savez(
        "fenics_concrete_thermo_mechanical",
        t=np.array(t_list),
        u=np.array(u_list),
        T=np.array(temperature_list),
        dof_map_u=dof_map_u,
        dof_map_t=dof_map_t,
        # E=E_sensor.data,
        # fc=fc_sensor.data,
        doh=doh_sensor.data,
    )


if __name__ == "__main__":
    test_hydration_with_body_forces(3)
