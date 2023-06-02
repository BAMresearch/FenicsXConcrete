import importlib
import inspect
import os
from abc import ABC

from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.sensor_definition.sensor_schema import generate_sensor_schema


def import_classes_from_module(module_name):
    module = importlib.import_module(module_name)
    class_names = set()

    module_path = os.path.dirname(module.__file__)
    for file_name in os.listdir(module_path):
        if file_name.endswith(".py") and file_name != "__init__.py":
            file_path = os.path.join(module_path, file_name)
            module_name = f"{module_name}.{os.path.splitext(file_name)[0]}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj is not ABC and issubclass(obj, BaseSensor):
                    class_names.add(obj.__name__)

    return class_names


def test_classes_in_dictionary():
    module_name = "fenicsxconcrete.sensor_definition"

    # Import all classes from the module
    classes = import_classes_from_module(module_name)
    dict_sensors_schema = generate_sensor_schema()

    # Check if the classes are present in the dictionary keys
    for cls in classes:
        assert cls in dict_sensors_schema["definitions"]
