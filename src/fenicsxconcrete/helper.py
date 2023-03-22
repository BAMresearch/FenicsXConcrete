from __future__ import annotations

from collections import UserDict  # because: https://realpython.com/inherit-python-dict/

import pint


class Parameters(UserDict):
    """
    A class that contains physical quantities for our model. Each new entry needs to be a pint quantity.
    """

    def __setitem__(self, key: str, value: pint.Quantity):
        assert isinstance(value, pint.Quantity)
        self.data[key] = value.to_base_units()

    def __add__(self, other: Parameters | None) -> Parameters:
        if other is None:
            dic = self
        else:
            dic = Parameters({**self, **other})
        return dic

    def to_magnitude(self) -> dict[str, int | str | float]:
        magnitude_dictionary = {}
        for key in self.keys():
            magnitude_dictionary[key] = self[key].magnitude

        return magnitude_dictionary
