from __future__ import annotations

from collections import UserDict  # because: https://realpython.com/inherit-python-dict/

import basix
import dolfinx as df
import numpy as np
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


class QuadratureRule:
    """
    An object that takes care of the creation of a quadrature rule and the creation of
    quadrature spaces.

    Args:
        type: The quadrature type. Examples are `basix.QuadratureType.Default`
            for Gaussian quadrature and `basix.QuadratureType.gll` for Gauss-Lobatto quadrature.
        cell_type: The type of FEM cell (`triangle, tetrahedron`,...).
        degree: The maximal degree that the quadrature rule should be able to integrate.


    Attributes:
        type (basix.QuadratureType): The quadrature type.
        cell_type (basix.CellType): The type of FEM cell.
        degree (int): The quadrature degree.
        points (np.ndarray): The quadrature points on the refernce cell.
        weights (np.ndarray): The weights of the quadrature rule.
        dx (ufl.measure): The appropriate measure for integrating ufl forms
            with the specified quadrature rule. **Always** use this measure
            when integrating a form that includes a quadrature function.

    """

    def __init__(
        self,
        type: basix.QuadratureType = basix.QuadratureType.Default,
        cell_type: basix.CellType = basix.CellType.triangle,
        degree: int = 1,
    ):
        self.type = type
        self.cell_type = cell_type
        self.degree = degree
        self.points, self.weights = basix.make_quadrature(self.quadrature_type, self.cell_type, self.degree)
        self.dx = ufl.dx(
            metadata={
                "quadrature_rule": self.quadrature_type.name,
                "quadrature_degree": self.degree,
            }
        )

    def create_quadrature_space(self, mesh: df.mesh.Mesh) -> df.fem.FunctionSpace:
        """
        Args:
            mesh: The mesh on which we want to create the space.

        Returns:
            A scalar quadrature `FunctionSpace` on `mesh`.
        """
        Qe = ufl.FiniteElement(
            "Quadrature",
            _basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_vector_space(self, mesh: df.mesh.Mesh, dim: int) -> df.fem.VectorFunctionSpace:
        """
        Args:
            mesh: The mesh on which we want to create the space.
            dim: The dimension of the vector at each dof.

        Returns:
            A vector valued quadrature `FunctionSpace` on `mesh`.
        """
        Qe = ufl.VectorElement(
            "Quadrature",
            _basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            dim=dim,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_tensor_space(self, mesh: df.mesh.Mesh, shape: tuple[int, int]) -> df.fem.TensorFunctionSpace:
        """
        Args:
            mesh: The mesh on which we want to create the space.
            shape: The shape of the tensor at each dof.

        Returns:
            A tensor valued quadrature `FunctionSpace` on `mesh`.
        """
        Qe = ufl.TensorElement(
            "Quadrature",
            _basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            shape=shape,
        )

        return df.fem.FunctionSpace(mesh, Qe)

    def number_of_points(self, mesh: df.mesh.Mesh) -> int:
        """
        Args:
            mesh: A mesh.
        Returns:
            Number of quadrature points that the QuadratureRule would generate on `mesh`
        """
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local
        return self.num_cells * self.weights.size

    def create_array(self, mesh: df.mesh.Mesh, shape: int | tuple[int, int] = 1) -> np.ndarray:
        """
        Creates array of a quadrature function without creating the function or the function space.
        This should be used, if operations on quadrature points are needed, but not all values are needed
        in a ufl form.

        Args:
            mesh: A mesh.
            shape: Local shape of the quadrature space. Example: `shape = 1` for Scalar,
              `shape = (n, 1)` for vector and `shape = (n,n)` for Tensor.
        Returns:
            An array that is equivalent to `quadrature_function.vector.array`.
        """
        n_points = self.number_of_points(mesh)
        n_local = shape if isinstance(shape, int) else shape[0] * shape[1]
        return np.zeros(n_points * n_local)


def _basix_cell_type_to_ufl(cell_type: basix.CellType) -> ufl.Cell:
    conversion = {
        basix.CellType.interval: ufl.interval,
        basix.CellType.triangle: ufl.triangle,
        basix.CellType.tetrahedron: ufl.tetrahedron,
        basix.CellType.quadrilateral: ufl.quadrilateral,
        basix.CellType.hexahedron: ufl.hexahedron,
    }
    return conversion[cell_type]


class QuadratureEvaluator:
    """
    A class that evaluates a ufl expression on a quadrature space.

    Args:
        ufl_expression: The ufl expression.
        mesh: The mesh on which we want to evaluate `ufl_expression`
        rule: The quadrature rule.
    """

    def __init__(self, ufl_expression: ufl.core.expr.Expr, mesh: df.mesh.Mesh, rule: QuadratureRule) -> None:
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local

        self.cells = np.arange(0, self.num_cells, dtype=np.int32)

        self.expr = df.fem.Expression(ufl_expression, self.rule.points)

    def evaluate(self, q: np.ndarray | df.fem.Function | None = None) -> np.ndarray | None:
        """
        Evaluate the expression.

        Args:
            q: The object we want to write the result to.

        Returns:
            A numpy array with the values if `q` is `None`, otherwise the result is written
            on `q` and `None` is returned.
        """
        if q is None:
            return self.expr.eval(self.cells)
        elif isinstance(q, np.ndarray):
            self.expr.eval(self.cells, values=q.reshape(self.num_cells, -1))
        elif isinstance(q, df.fem.Function):
            self.expr.eval(self.cells, values=q.vector.array.reshape(self.num_cells, -1))
            q.x.scatter_forward()


def project(
    v: df.fem.Function | ufl.core.expr.Expr, V: df.fem.FunctionSpace, dx: ufl.Measure, u: df.fem.Function | None = None
) -> None | df.fem.Function:
    """
    Calculates an approximation of `v` on the space `V`
    Args:
        v: The expression that we want to evaluate.
        V: The function space on which we want to evaluate.
        dx: The measure that is used for the integration. This is important, if we want to evaluate
            a Quadrature function on a _normal_ space.
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
