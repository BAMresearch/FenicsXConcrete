""" 
Compression cylinder setup with a linear elastic constitutive model
===================================================================

This experiment setup models a cylinder under compression with two options for the boundary conditions: free and fixed.
The material model used is a linear elastic constitutive model.
"""

# %%
# Parameters
# ----------
# The following parameters can be defined for this setup:
#
# * `E` (Young's modulus)
# * `nu` (Poisson's ratio)
# * `radius` (radius of the cylinder)
# * `height` (height of the cylinder)
# * `dim` (dimension of the problem) either 2 or 3
# * `degree` (degree of the finite element space) either 1 or 2
# * `bc_setting` (boundary condition setting) either 'free' or 'fixed'
# * `mesh_density` (mesh density) in 1/m
#
# The parameters must be defined as `pint` objects
# Parameters required but not defined are set to default values (from class function default_parameters).
#
# Example code
# ------------
# In this example, we define the parameters for the compression cylinder setup, set the displacement and sensor, and then create the experiment and the problem using `fenicxconcrete`.
# Finally, we solve the problem and get the measured reaction force at the bottom of the cylinder.

from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.util import ureg

parameters = {}
parameters["E"] = 1023 * ureg("MPa")
parameters["nu"] = 0.0 * ureg("")
parameters["radius"] = 0.006 * ureg("m")
parameters["height"] = 0.012 * ureg("m")
parameters["dim"] = 3 * ureg("")
parameters["degree"] = 2 * ureg("")
parameters["bc_setting"] = "fixed" * ureg("")
parameters["mesh_density"] = 10 * ureg("")

displacement = -0.003 * ureg("m")
sensor = ReactionForceSensor()

experiment = CompressionCylinder(parameters)
problem = LinearElasticity(experiment, parameters)
problem.add_sensor(sensor)

problem.experiment.apply_displ_load(displacement)

problem.solve()

measured = problem.sensors[sensor.name].data[0][-1]


print(f"The reaction force is {abs(measured):.2f} {sensor.units}.")
