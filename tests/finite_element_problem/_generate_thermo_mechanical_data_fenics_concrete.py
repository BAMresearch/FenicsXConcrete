"""
This file needs to run in an environment with version 
(commit 13486645b01665b4da248edd268b1904b0b5b745 (HEAD -> master, tag: v0.9.0, origin/master, origin/HEAD))
of FenicsConcrete https://github.com/BAMresearch/FenicsConcrete

This file should not run during tests. It is only in this directory in order to generate data that will be compared 
to the new implementation of the thermo mechanical model.
"""

try:
    import fenics_concrete
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """This file needs to run in an environment with version
        (commit 13486645b01665b4da248edd268b1904b0b5b745 (HEAD -> master, tag: v0.9.0, origin/master, origin/HEAD))
        of FenicsConcrete"""
    )
from pathlib import Path

import numpy as np

parameters = fenics_concrete.Parameters()  # using the current default values
# general
parameters["log_level"] = "WARNING"
# mesh
parameters["mesh_setting"] = "left/right"  # default boundary setting
parameters["dim"] = 3
parameters["mesh_density"] = 2
# temperature boundary
parameters["bc_setting"] = "full"  # default boundary setting
parameters["T_0"] = 20  # inital concrete temperature
parameters["T_bc1"] = 20  # temperature boundary value 1

parameters["density"] = 2350  # in kg/m^3 density of concrete
parameters["density_binder"] = 1440  # in kg/m^3 density of the binder
parameters["themal_cond"] = 2.0  # effective thermal conductivity, approx in Wm^-3K^-1, concrete!
# self.specific_heat_capacity = 9000  # effective specific heat capacity in J kg⁻1 K⁻1
parameters["vol_heat_cap"] = 2.4e6  # volumetric heat cap J/(m3 K)
parameters["b_ratio"] = 0.2  # volume percentage of binder
parameters["Q_pot"] = 500e3  # potential heat per weight of binder in J/kg
# p['Q_inf'] = self.Q_pot * self.density_binder * self.b_ratio  # potential heat per concrete volume in J/m3
parameters["B1"] = 2.916e-4  # in 1/s
parameters["B2"] = 0.0024229  # -
parameters["eta"] = 5.554  # something about diffusion
parameters["alpha_max"] = 0.87  # also possible to approximate based on equation with w/c
parameters["alpha_tx"] = 0.68  # also possible to approximate based on equation with w/c
parameters["E_act"] = 5653 * 8.3145  # activation energy in Jmol^-1
parameters["T_ref"] = 25  # reference temperature in degree celsius
# setting for temperature adjustment
parameters["temp_adjust_law"] = "exponential"
# polinomial degree
parameters["degree"] = 2  # default boundary setting
### paramters for mechanics problem
parameters["E"] = 42000000  # Youngs Modulus N/m2 or something...
parameters["nu"] = 0.2  # Poissons Ratio
# required paramters for alpha to E mapping
parameters["alpha_t"] = 0.2
parameters["alpha_0"] = 0.05
parameters["a_E"] = 0.6
# required paramters for alpha to tensile and compressive stiffness mapping
parameters["fc"] = 6210000
parameters["a_fc"] = 1.2
parameters["ft"] = 467000
parameters["a_ft"] = 1.0

experiment = fenics_concrete.ConcreteCubeExperiment(parameters)
problem = fenics_concrete.ConcreteThermoMechanical(experiment=experiment, parameters=parameters, vmapoutput=False)


E_sensor = fenics_concrete.sensors.YoungsModulusSensor((0.25, 0.25, 0.25))
fc_sensor = fenics_concrete.sensors.CompressiveStrengthSensor((0.25, 0.25, 0.25))
doh_sensor = fenics_concrete.sensors.DOHSensor((0.25, 0.25, 0.25))
# t_sensor = fenics_concrete.sensors.TemperatureSensor((0.25, 0.25))


problem.add_sensor(E_sensor)
problem.add_sensor(fc_sensor)
problem.add_sensor(doh_sensor)
# problem.add_sensor(t_sensor)

# data for time stepping
dt = 3600  # 60 min step

# set time step
problem.set_timestep(dt)  # for time integration scheme

# print(problem.p)
# initialize time
t = dt  # first time step time
t_list = []
u_list = []
temperature_list = []
doh = 0

# import matplotlib.pyplot as plt
# delta_alpha = np.linspace(0,0.006, 1000)

# plt.plot(delta_alpha,[problem.temperature_problem.delta_alpha_fkt(delta_alpha_i, 0., 293.15) for delta_alpha_i in delta_alpha])
# plt.show()

while doh < parameters["alpha_tx"]:  # time
    # solve temp-hydration-mechanics
    print("solving at t=", t)
    problem.solve(t=t)  # solving this
    t_list.append(t)
    u_list.append(problem.displacement.vector().get_local())
    temperature_list.append(problem.temperature.vector().get_local())
    # prepare next timestep
    t += dt
    # import sys
    # sys.exit()

    # get last measure value
    doh = problem.sensors[doh_sensor.name].data[-1]
dof_map_u = problem.displacement.function_space().tabulate_dof_coordinates()
dof_map_t = problem.temperature.function_space().tabulate_dof_coordinates()
np.savez(
    Path(__file__).parent / "fenics_concrete_thermo_mechanical.npz",
    t=np.array(t_list),
    u=np.array(u_list),
    T=np.array(temperature_list),
    dof_map_u=dof_map_u,
    dof_map_t=dof_map_t,
    E=E_sensor.data,
    fc=fc_sensor.data,
    doh=doh_sensor.data,
)
