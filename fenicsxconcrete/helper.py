import pint


class Parameters(dict):
    """
    A class that contains physical quantities for our model. Each new entry needs to be a pint quantity.
    """

    def __setitem__(self, key: str, value: pint.Quantity):
        assert isinstance(value, pint.Quantity)
        super().__setitem__(key, value.to_base_units())

    def __getattr__(self, key: str):
        return self[key]

    def __setattr__(self, key: str, value: pint.Quantity):
        assert isinstance(value, pint.Quantity)
        assert key in self
        self[key] = value

    def __add__(self, other):
        if other == None:
            dic = self
        else:
            dic = Parameters({**self, **other})
        return dic
