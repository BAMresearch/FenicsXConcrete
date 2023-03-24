# Three point bending experiment with linear elastic constitutive model

This example shows how to setup a three point bending beam and access the displacement data of a specific point.

The following functions are required
```python
from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.unit_registry import ureg
```
First, the setup is initialized, in this case a simply supported beam with a distributed load.
All parameters are `pint` objects.
The geometry, the mesh and the load need to be defined here.
```python
parameters = {}
parameters["length"] = 10 * ureg("m")
parameters["height"] = 0.5 * ureg("m")
parameters["width"] = 0.3 * ureg("m")  # only relevant for 3D case
parameters["dim"] = 3 * ureg("")
parameters["num_elements_length"] = 10 * ureg("")
parameters["num_elements_height"] = 3 * ureg("")
parameters["num_elements_width"] = 3 * ureg("")  # only relevant for 3D case
parameters["load"] = 200 * ureg("kN/m^2")

beam_setup = SimpleBeam(parameters)
```
Second, the linear elastic problem is initialized using the setup and further material paramters.
The same dictionary can be used for the material parameters.
```python
parameters["rho"] = 7750 * ureg("kg/m^3")
parameters["E"] = 210e9 * ureg("N/m^2")
parameters["nu"] = 0.28 * ureg("")

problem = LinearElasticity(beam_setup, parameters)
```
Third, a sensor is setup and added to access results of the FEM simulation.
```python
sensor_location = [parameters["length"].magnitude / 2, 0.0, 0.0]
sensor = DisplacementSensor([sensor_location])

problem.add_sensor(sensor)
```
Finally, the problem is solved and the sensor data accessed.
```python
problem.solve()

displacement_data = problem.sensors["DisplacementSensor"].data
```