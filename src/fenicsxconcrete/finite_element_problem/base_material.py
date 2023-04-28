from abc import ABC, abstractmethod
from pathlib import Path, PosixPath

import pint

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import LogMixin, Parameters
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.unit_registry import ureg


class MaterialProblem(ABC, LogMixin):
    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        pv_name: str = "pv_output_full",
        pv_path: PosixPath | None = None,
    ) -> None:
        """Base material problem.

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

        # setting up default material parameters
        default_fem_parameters = Parameters()
        default_fem_parameters["g"] = 9.81 * ureg("m/s^2")

        # adding experimental parameters to dictionary to combine to one
        default_fem_parameters.update(self.experiment.parameters)
        # update with input parameters
        default_fem_parameters.update(parameters)
        self.parameters = default_fem_parameters
        # remove units for use in fem model
        self.p = self.parameters.to_magnitude()
        self.experiment.p = self.p  # update experimental parameter list for use in e.g. boundary definition

        self.sensors = self.SensorDict()  # list to hold attached sensors

        # settin gup path for paraview output
        if not pv_path:
            pv_path = "."
        self.pv_output_file = Path(pv_path) / (pv_name + ".xdmf")

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

    @abstractmethod
    def setup(self) -> None:
        # initialization of this specific problem
        """Implemented in child if needed"""

    @abstractmethod
    def solve(self) -> None:
        # define what to do, to solve this problem
        """Implemented in child if needed"""

    @abstractmethod
    def compute_residuals(self) -> None:
        # define what to do, to compute the residuals. Called in solve
        """Implemented in child if needed"""

    def add_sensor(self, sensor: BaseSensor) -> None:
        if isinstance(sensor, BaseSensor):
            self.sensors[sensor.name] = sensor
        else:
            raise ValueError("The sensor must be of the class Sensor")

    def clean_sensor_data(self) -> None:
        for sensor_object in self.sensors.values():
            sensor_object.data.clear()

    def delete_sensor(self) -> None:
        del self.sensors
        self.sensors = self.SensorDict()

    class SensorDict(dict):
        """
        Dict that also allows to access the parameter p["parameter"] via the matching attribute p.parameter
        to make access shorter

        When to sensors with the same name are defined, the next one gets a number added to the name
        """

        def __getattr__(self, key):
            return self[key]

        def __setitem__(self, initial_key: str, value: BaseSensor) -> None:
            # check if key exists, if so, add a number to the name
            i = 2
            key = initial_key
            if key in self:
                while key in self:
                    key = initial_key + str(i)
                    i += 1
                # rename the sensor object
                value.name = key

            super().__setitem__(key, value)
