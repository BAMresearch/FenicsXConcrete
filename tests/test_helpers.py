import dolfinx as df
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from pint import UnitRegistry

from fenicsxconcrete.util import Parameters, QuadratureEvaluator, QuadratureRule, project

ureg = UnitRegistry()


def test_parameters() -> None:
    parameters = Parameters()
    parameters["length"] = 42.0 * ureg.cm

    # Check if units are converted correctly
    assert parameters["length"].units == ureg.meter

    parameters_2 = Parameters()
    parameters_2["temperature"] = 2.0 * ureg.kelvin

    parameters_combined = parameters + parameters_2
    keys = parameters_combined.keys()
    assert "length" in keys and "temperature" in keys
    assert (
        parameters_combined["length"] == parameters["length"]
        and parameters_combined["temperature"] == parameters_2["temperature"]
    )


def test_parameter_dic_functions() -> None:
    parameters = Parameters()
    # testing if adding None to dictionary works
    new = parameters + None
    assert new is parameters


def test_parameter_dic_update() -> None:
    parameters = Parameters()

    # testing that update still requires a pint object
    p_wo_pint = {"length": 0.006}
    with pytest.raises(AssertionError):
        parameters.update(p_wo_pint)

    # testing that conversion to base units works with update
    length = 6000
    p_with_pint = {"length": length * ureg("mm")}
    parameters.update(p_with_pint)

    assert parameters["length"].magnitude == length / 1000


# @pytest.mark.parametrize("dim", [2, 3])
def test_project() -> None:
    mesh = df.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)
    P1 = df.fem.FunctionSpace(mesh, ("P", 1))
    u = df.fem.Function(P1)
    v = df.fem.Function(P1)
    u.interpolate(lambda x: x[0] + x[1] + x[2])
    project(u, P1, ufl.dx, v)
    assert np.linalg.norm(u.vector.array - v.vector.array) / np.linalg.norm(u.vector.array) < 1e-4


def test_quadrature_rule() -> None:
    """check if all spaces and arrays are compatible with each other"""
    rule = QuadratureRule()
    mesh = df.mesh.create_unit_square(MPI.COMM_SELF, 2, 2)

    lagrange_space = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 2))
    v = df.fem.Function(lagrange_space)

    v.interpolate(lambda x: (42.0 * x[0], 16.0 * x[1]))

    strain_form = ufl.sym(ufl.grad(v))
    strain_evaluator = QuadratureEvaluator(strain_form, mesh, rule)

    q_space = rule.create_quadrature_space(mesh)
    q_function = df.fem.Function(q_space)
    q_array = rule.create_quadrature_array(mesh, 1)

    assert q_function.vector.array.shape == q_array.shape

    q_vector_space = rule.create_quadrature_vector_space(mesh, 6)
    q_vector_function = df.fem.Function(q_vector_space)
    q_vector_array = rule.create_quadrature_array(mesh, 6)

    assert q_vector_function.vector.array.shape == q_vector_array.shape

    q_tensor_space = rule.create_quadrature_tensor_space(mesh, (2, 2))
    q_tensor_function = df.fem.Function(q_tensor_space)
    q_tensor_array = rule.create_quadrature_array(mesh, (2, 2))

    assert q_tensor_function.vector.array.shape == q_tensor_array.shape

    assert 6 * q_function.vector.array.size == q_vector_function.vector.array.size
    assert 4 * q_function.vector.array.size == q_tensor_function.vector.array.size

    # check if project and QuadratureEvaluator give the same result
    project(strain_form, q_tensor_space, rule.dx, q_tensor_function)

    assert (
        np.linalg.norm(q_tensor_function.vector.array - strain_evaluator.evaluate().flatten())
        / np.linalg.norm(q_tensor_function.vector.array)
        < 1e-12
    )
