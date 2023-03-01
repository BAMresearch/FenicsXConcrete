#import sys
#print(sys.path)
import dolfinx as df
import ufl
from petsc4py.PETSc import ScalarType
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import Parameters
import numpy as np

# this is necessary, otherwise this warning will not stop
# https://fenics.readthedocs.io/projects/ffc/en/latest/_modules/ffc/quadrature/deprecation.html
import warnings
#from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
#df.parameters["form_compiler"]["representation"] = "quadrature"
#warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
from fenicsxconcrete.unit_registry import ureg


class linear_elasticity(MaterialProblem):
    """Material definition for linear elasticity"""

    def __init__(self, experiment=None, parameters=None, pv_name='pv_output_linear_elasticity'):
        """Initializes the object by calling super().__init__

        Parameters
        ----------
            experiment : object, optional
                When no experiment is passed, the dummy experiment "MinimalCubeExperiment" is added
            parameters : dictionary, optional
                Dictionary with parameters. When none is provided, default values are used
            pv_name : string, optional
                Name of the paraview file, if paraview output is generated
        """
        # generate "dummy" experiment when none is passed

        #if experiment is None:
        #    #experiment = experimental_setups.MinimalCubeExperiment(parameters)

        super().__init__(experiment, parameters, pv_name)

    def setup(self):
        default_p = Parameters()

        self.parameters = default_p + self.parameters
          
        self.residual = None  # initialize residual


        self.p = self.parameters.to_magnitude()
        # Constant E and nu fields.
        E = self.p['E']
        nu = self.p['nu']

        self.lambda_ = df.fem.Constant(self.experiment.mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        self.mu = df.fem.Constant(self.experiment.mesh, E / (2.0 * (1.0 + nu)))
        
        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.experiment.V)
        self.v = ufl.TestFunction(self.experiment.V)
        
        # Selects the problem which you want to solve
        if self.p['problem'] == 'tensile_test':
            self.T = df.fem.Constant(self.experiment.mesh, ScalarType((self.p['load'], 0)))
            ds = self.experiment.create_neumann_boundary()
            self.L =  ufl.dot(self.T, self.v) * ds(1) 

        elif self.p['problem']== 'cantilever_beam':
            if self.p['dim'] == 2:
                #f = df.Constant((0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, -self.p['rho']*self.p['g'])))
            elif self.p['dim'] == 3:
                #f = df.Constant((0, 0, 0))
                f = df.fem.Constant(self.experiment.mesh, ScalarType((0, 0, -self.p['rho']*self.p['g'])))
            else:
                raise Exception(f'wrong dimension {self.p["dim"]} for problem setup')
                
            self.L =  ufl.dot(f, self.v) * ufl.dx
        else:
            exit()

        self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
        self.weak_form_problem = df.fem.petsc.LinearProblem(self.a,
                                                            self.L,
                                                            bcs=self.experiment.bcs,
                                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


    # Stress computation for linear elastic problem 
    def epsilon(self, u):
        return ufl.sym(ufl.grad(u)) 


    #Deterministic
    def sigma(self, u):
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(self.p['dim']) + 2*self.mu*self.epsilon(u)

    def solve(self, t=1.0):        
        self.displacement = self.weak_form_problem.solve()
        #self.stress = self.sigma(self.displacement)

        # TODO make some switch in sensor definition to trigger this...
        #self.compute_residual()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)
            
    #def compute_residual(self):
    #    # compute reaction forces
    #    self.residual = df.action(self.a, self.displacement) - self.L

    def pv_plot(self, t=0):
        # paraview output
        
        # Displacement Plot
        with df.io.XDMFFile(self.experiment.mesh.comm, "Displacement.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.experiment.mesh)
            xdmf.write_function(self.displacement)

        #Stress Plot
        #with df.io.XDMFFile(self.experiment.mesh.comm, "Stress.xdmf", "w") as xdmf:
        #    xdmf.write_mesh(self.experiment.mesh.comm)
        #    xdmf.write_function(self.stress)


