import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor


def test_sensor_dict() -> None:
    """This function tests the sensor dict"""

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    problem = LinearElasticity(default_experiment, fem_parameters)

    sensor_location = [0.0, 0.0, 0.0]
    sensor = DisplacementSensor([sensor_location])

    # testing the renaming of multiple sensors
    problem.add_sensor(sensor)
    problem.add_sensor(sensor)
    assert "DisplacementSensor2" in problem.sensors.keys()

    # accessing as attribute
    assert problem.sensors.DisplacementSensor2


def test_sensor_error() -> None:
    """This function tests the add sensor function"""

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    problem = LinearElasticity(default_experiment, fem_parameters)

    with pytest.raises(ValueError):
        problem.add_sensor("not a sensor")


def test_sensor_options() -> None:
    """This function tests the function of creating and deleting sensors"""

    # setting up problem
    setup_parameters = CantileverBeam.default_parameters()
    default_setup, fem_parameters = LinearElasticity.default_parameters()

    sensor_location = [setup_parameters["length"].magnitude, 0.0, 0.0]
    sensor = DisplacementSensor(sensor_location)

    # setting up the problem
    experiment = CantileverBeam(setup_parameters)  # Specifies the domain, discretises it and apply Dirichlet BCs
    problem = LinearElasticity(experiment, fem_parameters)

    # check that no sensors yet exist
    assert problem.sensors == {}

    # add sensor
    problem.add_sensor(sensor)

    # check that sensor exists
    assert problem.sensors != {}

    # check that no data is in sensor
    assert problem.sensors[sensor.name].data == []

    # solving and plotting
    problem.solve()

    # check that some data is in sensor
    measure = deepcopy(problem.sensors[sensor.name].data)
    assert measure != []

    # check export sensor data
    problem.export_sensors_metadata(Path("sensors_metadata.json"))
    expected_metadata = {
        "sensors": [
            {
                "id": "DisplacementSensor",
                "type": "DisplacementSensor",
                "sensor_file": "displacement_sensor",
                "units": "meter",
                "dimensionality": "[length]",
                "where": [1, 0.0, 0.0],
            }
        ]
    }
    with open("sensors_metadata.json", "r") as f:
        sensor_metadata = json.load(f)
    assert sensor_metadata == expected_metadata

    # check cleaning of sensor data
    problem.clean_sensor_data()
    assert problem.sensors[sensor.name].data == []

    # delete sensor
    problem.delete_sensor()
    assert problem.sensors == {}

    # check import sensor data
    problem.import_sensors_from_metadata(Path("sensors_metadata.json"))

    os.remove("sensors_metadata.json")

    # repeat solving and plotting
    problem.solve()

    # repeat check that some data is in imported sensor
    assert problem.sensors[sensor.name].data[0] == pytest.approx(measure[0])
