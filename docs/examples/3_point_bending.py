"""
Three point bending experiment with a linear elastic constitutive model
=======================================================================

This example demonstrates how to set up a three-point bending beam and access
the displacement data of a specific point using a linear elastic constitutive model.
"""
# %%
# To run this example, the following functions need to be imported:

from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.unit_registry import ureg

# %%
# Setting up the Beam
# -------------------
# First, initialize the setup for the simply supported beam with a distributed load.
# Define the geometry, the mesh, and the load.
# Note, all parameters must be pint objects:

parameters = {}
parameters["length"] = 10 * ureg("m")
parameters["height"] = 0.5 * ureg("m")
parameters["width"] = 0.3 * ureg("m")
parameters["dim"] = 3 * ureg("")
parameters["num_elements_length"] = 10 * ureg("")
parameters["num_elements_height"] = 3 * ureg("")
parameters["num_elements_width"] = 3 * ureg("")
parameters["load"] = 200 * ureg("kN/m^2")

beam_setup = SimpleBeam(parameters)

# %%
# Initializing the Linear Elasticity Problem
# ------------------------------------------
# Second, initialize the linear elastic problem using the setup object and further material parameters:

parameters["rho"] = 7750 * ureg("kg/m^3")
parameters["E"] = 210e9 * ureg("N/m^2")
parameters["nu"] = 0.28 * ureg("")

problem = LinearElasticity(beam_setup, parameters)

# %%
# Setting Up the Sensor
# ---------------------
# Third, set up a sensor and add it to the problem to access results of the FEM simulation:

sensor_location = [parameters["length"].magnitude / 2, parameters["width"].magnitude / 2, 0.0]
sensor = DisplacementSensor(sensor_location)

problem.add_sensor(sensor)

# %%
# Solving the Problem and Accessing the Results
# ---------------------------------------------
# Finally, solve the problem and access the sensor data:
problem.solve()

displacement_data = problem.sensors["DisplacementSensor"].data[0]

print(f"The displacement at the center of the beam in loading direction is {displacement_data[2]:.2f} {sensor.units}.")
