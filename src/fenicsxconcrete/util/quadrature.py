import basix
import dolfinx as df
import numpy as np
import ufl


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
        cell_type (ufl.Cell): The type of FEM cell.
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
        cell_type: ufl.Cell = ufl.triangle,
        degree: int = 1,
    ):
        self.type = type
        self.cell_type = cell_type
        self.degree = degree
        basix_cell = _ufl_cell_type_to_basix(self.cell_type)
        self.points, self.weights = basix.make_quadrature(self.type, basix_cell, self.degree)
        self.dx = ufl.dx(
            metadata={
                "quadrature_rule": self.type.name,
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
        assert mesh.ufl_cell() == self.cell_type
        Qe = ufl.FiniteElement(
            "Quadrature",
            self.cell_type,
            self.degree,
            quad_scheme=self.type.name,
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
        assert mesh.ufl_cell() == self.cell_type
        Qe = ufl.VectorElement(
            "Quadrature",
            self.cell_type,
            self.degree,
            quad_scheme=self.type.name,
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
        assert mesh.ufl_cell() == self.cell_type
        Qe = ufl.TensorElement(
            "Quadrature",
            self.cell_type,
            self.degree,
            quad_scheme=self.type.name,
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
        assert mesh.ufl_cell() == self.cell_type

        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local
        return self.num_cells * self.weights.size

    def create_quadrature_array(self, mesh: df.mesh.Mesh, shape: int | tuple[int, int] = 1) -> np.ndarray:
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


def _ufl_cell_type_to_basix(cell_type: ufl.Cell) -> basix.CellType:
    conversion = {
        ufl.interval: basix.CellType.interval,
        ufl.triangle: basix.CellType.triangle,
        ufl.tetrahedron: basix.CellType.tetrahedron,
        ufl.quadrilateral: basix.CellType.quadrilateral,
        ufl.hexahedron: basix.CellType.hexahedron,
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
        assert mesh.ufl_cell() == rule.cell_type
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local

        self.cells = np.arange(0, self.num_cells, dtype=np.int32)

        self.expr = df.fem.Expression(ufl_expression, rule.points)

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
