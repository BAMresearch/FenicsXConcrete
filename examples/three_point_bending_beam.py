import os
import pytest
from pathlib import Path

from fenicsxconcrete.unit_registry import ureg
from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor



# # setup paths and directories
data_dir = 'data_files'
data_path = Path(__file__).parent / data_dir
#
# define file name and path for paraview output
file_name = f'3pointBendingExample'

setup_parameters = {}
setup_parameters['length'] = 10 * ureg('m')
setup_parameters['height'] = 0.5 * ureg('m')
setup_parameters['width'] = 0.3 * ureg('m')  # only relevant for 3D case
setup_parameters['dim'] = 2 * ureg('')
setup_parameters['num_elements_length'] = 20 * ureg('')
setup_parameters['num_elements_height'] = 4 * ureg('')
setup_parameters['num_elements_width'] = 3 * ureg('')  # only relevant for 3D case

fem_parameters = {}
fem_parameters['rho'] = 7750 * ureg('kg/m^3')
fem_parameters['E'] = 210e9 * ureg('N/m^2')
fem_parameters['nu'] = 0.28 * ureg('')

# Defining sensor positions
sensor_location = [setup_parameters['length'].magnitude/2, setup_parameters['width'].magnitude/2, 0.0]
sensor = DisplacementSensor([sensor_location])

# setting up the problem
experiment = SimpleBeam(setup_parameters)
problem = LinearElasticity(experiment, fem_parameters, pv_name=file_name, pv_path=str(data_path))
problem.add_sensor(sensor)

    # solving and plotting
problem.solve()
problem.pv_plot()
displacement_data = problem.sensors['DisplacementSensor'].data[-1]
print(displacement_data)
# TODO stress sensor!!!

