import dolfinx as df
from pathlib import Path
import sys
from loguru import logger
import logging
from abc import ABC, abstractmethod
import pint
from fenicsxconcrete.experimental_setup.base_experiment import Experiment

from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.sensor_definition.base_sensor import Sensors

class MaterialProblem(ABC):
    def __init__(self,
                 experiment,
                 parameters: dict[str, pint.Quantity],
                 pv_name='pv_output_full',
                 pv_path=None):
        """"base material problem

        Parameters
        ----------
            experiment : object
            parameters : dictionary, optional
                Dictionary with parameters. When none is provided, default values are used
            pv_name : string, optional
                Name of the paraview file, if paraview output is generated
            pv_path : string, optional
                Name of the paraview path, if paraview output is generated
        """

        self.experiment = experiment
        self.mesh = self.experiment.mesh
        # setting up paramters
        # adding experimental parameters to material parameters
        parameters.update(self.experiment.parameters)
        self.parameters = parameters
        # remove units for use in fem model
        self.p = self.parameters.to_magnitude()
        self.experiment.p = self.p  # update experimental parameter list for use in e.g. boundary definition

        # set log level...
        if self.p['log_level'] == 'NOTSET':
            df.log.LogLevel(0)
            logging.getLogger("FFC").setLevel(logging.NOTSET)
            logging.getLogger("UFL").setLevel(logging.NOTSET)
            logger.add(sys.stderr, level="NOTSET")
        elif self.p['log_level'] == 'DEBUG':
            df.log.LogLevel(10)
            logging.getLogger("FFC").setLevel(logging.DEBUG)
            logging.getLogger("UFL").setLevel(logging.DEBUG)
            logger.add(sys.stderr, level="DEBUG")
        elif self.p['log_level'] == 'INFO':
            df.log.LogLevel(20)
            logging.getLogger("FFC").setLevel(logging.INFO)
            logging.getLogger("UFL").setLevel(logging.INFO)
            logger.add(sys.stderr, level="INFO")
        elif self.p['log_level'] == 'WARNING':
            df.log.LogLevel(30)
            logging.getLogger("FFC").setLevel(logging.WARNING)
            logging.getLogger("UFL").setLevel(logging.WARNING)
            logger.add(sys.stderr, level="WARNING")
        elif self.p['log_level'] == 'ERROR':
            df.log.LogLevel(40)
            logging.getLogger("FFC").setLevel(logging.ERROR)
            logging.getLogger("UFL").setLevel(logging.ERROR)
            logger.add(sys.stderr, level="ERROR")
        elif self.p['log_level'] == 'CRITICAL':
            df.log.LogLevel(50)
            logging.getLogger("FFC").setLevel(logging.CRITICAL)
            logging.getLogger("UFL").setLevel(logging.CRITICAL)
            logger.add(sys.stderr, level="CRITICAL")
        else:
            level = self.p['log_level']
            raise Exception(f'unknown log level {level}')

        self.sensors = Sensors()  # list to hold attached sensors

        # settin gup path for paraview output
        if not pv_path:
            pv_path = "."
        self.pv_output_file = Path(pv_path) / (pv_name + '.xdmf')

        # setup fields for sensor output, can be defined in model
        self.displacement = None
        self.temperature = None
        self.degree_of_hydration = None
        self.q_degree_of_hydration = None

        self.residual = None  # initialize residual

        # setup the material object to access the function
        self.setup()

    @staticmethod
    @abstractmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """returns a dictionary with required parameters and a set of working values as example"""
        # this must de defined in each setup class
        pass

    @abstractmethod
    def setup(self):
        # initialization of this specific problem
        pass

    @abstractmethod
    def solve(self):
        # define what to do, to solve this problem
        pass

    def add_sensor(self, sensor):
        self.sensors[sensor.name] = sensor

    def clean_sensor_data(self):
        for sensor_object in self.sensors.values():
            sensor_object.data.clear()

    def delete_sensor(self):
        del self.sensors
        self.sensors = Sensors()
