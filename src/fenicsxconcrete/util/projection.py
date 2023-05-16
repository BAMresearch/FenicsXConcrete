import dolfinx as df
import ufl


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
