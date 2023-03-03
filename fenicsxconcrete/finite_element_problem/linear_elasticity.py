import dolfinx as df
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem


class LinearElasticity(MaterialProblem):
    """Material definition for linear elasticity"""

    def __init__(self, experiment, parameters, pv_name='pv_output_linear_elasticity', pv_path=None):
        """defines default parameters, for the rest, see base class"""

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()
        default_p['dummy'] = 42 * ureg('')   # just an example, maybe useful for other material classes?

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(experiment, default_p, pv_name, pv_path)

    def setup(self):
        # compute different set of elastic moduli
        self.lambda_ = df.fem.Constant(self.mesh,
                                       self.p['E'] * self.p['nu'] / ((1 + self.p['nu']) * (1 - 2 * self.p['nu'])))
        self.mu = df.fem.Constant(self.mesh, self.p['E'] / (2 * (1 + self.p['nu'])))

        # define function space ets.
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", self.p['degree'])) # 2 for quadratic elements
        self.V_scalar = df.fem.FunctionSpace(self.mesh, ("Lagrange", self.p['degree']))

        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # initialize L field, not sure if this is the best way...
        zero_field = df.fem.Constant(self.mesh, ScalarType(np.zeros(self.p['dim'])))
        self.L = ufl.dot(zero_field, self.v) * ufl.dx

        # apply external loads
        external_force = self.experiment.create_force_boundary(self.v)
        if external_force:
            self.L = self.L + external_force

        body_force = self.experiment.create_body_force(self.v)
        if body_force:
            self.L = self.L + body_force

        # boundary conditions only after function space
        bcs = self.experiment.create_displacement_boundary(self.V)

        self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
        self.weak_form_problem = df.fem.petsc.LinearProblem(self.a,
                                                            self.L,
                                                            bcs=bcs,
                                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 

    def sigma(self, u):
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(self.p['dim']) + 2 * self.mu * self.epsilon(u)

    def solve(self, t=1.0):        
        self.displacement = self.weak_form_problem.solve()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

    # paraview output
    # TODO move this to sensor definition!?!?!
    def pv_plot(self, t=0):
        # TODO add possibility for multiple time steps???
        # Displacement Plot

        #"Displacement.xdmf"
        #pv_output_file
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.displacement)
