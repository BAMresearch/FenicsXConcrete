from __future__ import annotations

import logging
from collections import UserDict  # because: https://realpython.com/inherit-python-dict/

import dolfinx as df
import pint
import ufl


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


class LogMixin(object):
    @property
    def logger(self):
        name = self.__class__.__module__
        return logging.getLogger(name)


def project(
    v: df.fem.Function | ufl.core.expr.Expr, V: df.fem.FunctionSpace, dx: ufl.Measure, u: df.fem.Function | None = None
) -> None | df.fem.Function:
    """
    Calculates an approximation of `v` on the space `V`

    Args:
        v: The expression that we want to evaluate.
        V: The function space on which we want to evaluate.
        dx: The measure that is used for the integration. This is important, if we want to evaluate
        a Quadrature function on a normal space.
        u: The output function.

    Returns:
        A function if `u` is None, otherwise `None`.
    """
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    a_proj = ufl.inner(dv, v_) * dx
    b_proj = ufl.inner(v, v_) * dx
    if u is None:
        solver = df.fem.petsc.LinearProblem(a_proj, b_proj)
        uh = solver.solve()
        return uh
    else:
        solver = df.fem.petsc.LinearProblem(a_proj, b_proj, u=u)
        solver.solve()
