import os, sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)
from fenicsxconcrete.sensor_definition import displacement_sensor
from fenicsxconcrete.experimental_setup.steel_beam_experiment import steel_beam
from fenicsxconcrete.finite_element_problem.linear_elastic_material import linear_elasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg

import numpy as np

def simple_setup(p, sensor):
    parameters = Parameters()  # using the current default values

    #parameters['log_level'] = 'WARNING'
    parameters['bc_setting'] = 'free' * ureg('')
    parameters['mesh_density'] = 10 * ureg('')

    parameters = parameters + p

    experiment = steel_beam(parameters)         # Specifies the domain, discretises it and apply Dirichlet BCs

    problem = linear_elasticity(experiment, parameters)      # Specifies the material law and weak forms.

    for i in range(len(sensor)):
        problem.add_sensor(sensor[i])

    problem.solve()  

    problem.pv_plot()

    return problem.sensors


p = Parameters()  # using the current default values
p['problem'] = 'cantilever_beam' * ureg('') #'cantilever_beam' #

# N/m², m, kg, sec, N
p['rho'] = 7750 * ureg('kg/m^3')
p['g'] = 9.81 * ureg('m/s^2')
p['E'] = 210e9 * ureg('N/m^2')
p['length'] = 1 * ureg('m')
p['breadth'] = 0.2 * ureg('m')
#p['load'] = 1000#-10e8

# MPa, mm, kg, sec, N
#p['rho'] = 7750e-9 #kg/mm³
#p['g'] = 9.81#e3 #mm/s² for units to be consistent g must be given in m/s².
#p['E'] = 210e3 #N/mm² or MPa
#p['length'] = 1000
#p['breadth'] = 200
#p['load'] = 100e-6 #N/mm²

p['nu'] = 0.28 * ureg('')
p['num_elements_length'] = 30 * ureg('')
p['num_elements_breadth'] = 20 * ureg('')
p['dim'] = 2 * ureg('')
#Defining sensor positions
sensor = []
sensor_pos_x = []
number_of_sensors = 20
for i in range(number_of_sensors):
    sensor.append(displacement_sensor.DisplacementSensor(np.array([[p['length'].magnitude/20*(i+1),
                                                                                    0.5*p['breadth'].magnitude,
                                                                                    0]])))
    sensor_pos_x.append(p['length'].magnitude/20*(i+1))

# Synthetic data generation
solution = simple_setup(p, sensor)
number_of_sensors = 20

def collect_sensor_solutions(model_solution, total_sensors):
    counter=0
    disp_model = np.zeros((total_sensors,2))
    for i in model_solution:
        disp_model[counter] = model_solution[i].data[-1]
        counter += 1
    return disp_model
    #print(measured[i].data[-1])

displacement_data = collect_sensor_solutions(solution, number_of_sensors)

""" import plotly.express as px
fig = px.line(x=sensor_pos_x, y=displacement_data[:,1])
fig.update_layout(
    title_text='Vertical Displacement Curve'
)
fig.show() """
# #
# import matplotlib.pyplot as plt
# #import numpy as np
# plt.plot(sensor_pos_x, displacement_data[:,1])
# plt.show()
