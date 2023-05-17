import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PosixPath

import dolfinx as df
import jsonschema
import pint
import ufl

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.sensor_definition.sensor_schema import generate_sensor_schema
from fenicsxconcrete.util import LogMixin, Parameters, ureg


@dataclass
class SolutionFields:
    """
    A dataclass to hold the solution fields of the problem.
    The list of names should be extendend when needed.

    Examples:
        Since this is a dataclass, the __init__ method is automatically
        generated and can be used to selectively set fields. All fields that
        are not explicitely set are set to their default value (here None).

        >>> fields = SolutionFields(displacement=some_function, temperature=some_other_function)
    """

    displacement: df.fem.Function | None = None
    velocity: df.fem.Function | None = None
    temperature: df.fem.Function | None = None
    nonlocal_strain: df.fem.Function | None = None


@dataclass
class QuadratureFields:
    """
    A dataclass to hold the quadrature fields (or ufl expressions)
    of the problem, at least those that we want to plot in paraview.
    Additionally, the measure for the integration and the type of function
    space is stored. The list of names should be extendend when needed.

    Examples:
        Since this is a dataclass, the __init__ method is automatically
        generated and can be used to selectively set fields. All fields that
        are not explicitely set are set to their default value (here None).

        >>> q_fields = QuadratureFields(measure=rule.dx, plot_space_type=("Lagrange", 4), stress=some_function)
    """

    measure: ufl.Measure | None = None
    plot_space_type: tuple[str, int] = ("DG", 0)
    mandel_stress: ufl.core.expr.Expr | df.fem.Function | None = None
    mandel_strain: ufl.core.expr.Expr | df.fem.Function | None = None
    stress: ufl.core.expr.Expr | df.fem.Function | None = None
    strain: ufl.core.expr.Expr | df.fem.Function | None = None
    degree_of_hydration: ufl.core.expr.Expr | df.fem.Function | None = None
    damage: ufl.core.expr.Expr | df.fem.Function | None = None


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
        self.fields = None
        self.q_fields = None

        self.residual = None  # initialize residual

        # set up xdmf file with mesh info
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w") as f:
            f.write_mesh(self.mesh)

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

    def export_sensors_metadata(self, path: Path) -> None:
        """Exports sensor metadata to JSON file according to the appropriate schema.

        Args:
            path : Path
                Path where the metadata should be stored

        """

        sensors_metadata_dict = {"sensors": []}

        for key, value in self.sensors.items():
            sensors_metadata_dict["sensors"].append(value.report_metadata())
            # sensors_metadata_dict[key]["name"] = key

        with open(path, "w") as f:
            json.dump(sensors_metadata_dict, f)

    def import_sensors_from_metadata(self, path: Path) -> None:
        """Import sensor metadata to JSON file and validate with the appropriate schema.

        Args:
            path : Path
                Path where the metadata file is

        """

        # Load and validate
        sensors_metadata_dict = {}
        with open(path, "r") as f:
            sensors_metadata_dict = json.load(f)
        schema = generate_sensor_schema()
        jsonschema.validate(instance=sensors_metadata_dict, schema=schema)

        for sensor in sensors_metadata_dict["sensors"]:
            # Dynamically import the module containing the class
            module_name = "fenicsxconcrete.sensor_definition." + sensor["sensor_file"].lower()
            module = importlib.import_module(module_name)

            # Create a dictionary of keyword arguments from the remaining properties in the dictionary
            kwargs = {
                k: v for k, v in sensor.items() if k not in ["id", "type", "sensor_file", "units", "dimensionality"]
            }

            # Dynamically retrieve the class by its name
            class_name = sensor["type"]
            MySensorClass = getattr(module, class_name)

            # Instantiate an object of the class with the given properties
            sensor_i = MySensorClass(name=sensor["id"], **kwargs)
            sensor_i.set_units(units=sensor["units"])

            self.add_sensor(sensor_i)

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
