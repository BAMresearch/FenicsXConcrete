def generate_sensor_schema() -> dict:
    """Function that returns the sensor schema. Necessary to include the schema in the package accessible.

    Returns:
      Schema for sensor's list metadata
    """

    schema = {
        "$schema": "http://json-schema.org/2020-12/schema#",
        "title": "SensorsList",
        "type": "object",
        "properties": {
            "sensors": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"$ref": "#/definitions/BaseSensor"},
                        {"$ref": "#/definitions/PointSensor"},
                        {"$ref": "#/definitions/DisplacementSensor"},
                        {"$ref": "#/definitions/TemperatureSensor"},
                        {"$ref": "#/definitions/ReactionForceSensor"},
                        {"$ref": "#/definitions/StrainSensor"},
                        {"$ref": "#/definitions/StressSensor"},
                        {"$ref": "#/definitions/YoungsModulusSensor"},
                        {"$ref": "#/definitions/DOHSensor"},
                    ]
                },
            }
        },
        "definitions": {
            "baseSensorProperties": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "A unique identifier for the sensor"},
                    "type": {"type": "string", "description": "The python class for the sensor"},
                    "units": {"type": "string", "description": "The unit of measurement for the sensor"},
                    "dimensionality": {
                        "type": "string",
                        "description": "The dimensionality of measurement for the sensor between brackets []",
                    },
                    "sensor_file": {
                        "type": "string",
                        "description": "Python file where the sensor is defined whithout extension",
                    },
                },
                "required": ["id", "type", "units", "dimensionality"],
            },
            "pointSensorProperties": {
                "allOf": [
                    {"$ref": "#/definitions/baseSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"where": {"type": "array", "description": "Location of the sensor"}},
                        "required": ["where"],
                    },
                ]
            },
            "BaseSensor": {
                "allOf": [
                    {"$ref": "#/definitions/baseSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "BaseSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "PointSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "PointSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "DisplacementSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "DisplacementSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "TemperatureSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "TemperatureSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "ReactionForceSensor": {
                "allOf": [
                    {"$ref": "#/definitions/baseSensorProperties"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "ReactionForceSensor", "description": "The type of sensor"},
                            "surface": {
                                "type": "string",
                                "description": "Surface where the reactionforce is measured",
                            },
                        },
                        "required": ["type", "surface"],
                    },
                ]
            },
            "StrainSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "StrainSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "StressSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "StressSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
            "YoungsModulusSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "YoungsModulusSensor", "description": "The type of sensor"},
                        },
                        "required": ["type"],
                    },
                ]
            },
            "DOHSensor": {
                "allOf": [
                    {"$ref": "#/definitions/pointSensorProperties"},
                    {
                        "type": "object",
                        "properties": {"type": {"const": "DOHSensor", "description": "The type of sensor"}},
                        "required": ["type"],
                    },
                ]
            },
        },
    }
    return schema
