from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
import dolfinx as df
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from fenicsxconcrete.unit_registry import ureg

class CantileverBeam(Experiment):
    """Sets up a cantilever beam, clamped on one side and loaded with gravity

    Attributes:
        see base class
    """

    def __init__(self, parameters):
        """defines default parameters, for the rest, see base class"""

        # initialize default parameters for the setup
        default_p = Parameters()
        default_p['degree'] = 2 * ureg('')  # polynomial degree
        default_p['load'] = 0 * ureg('N')  # TODO: find a better way

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(default_p)

    def setup(self):
        """defines the mesh for 2D or 3D"""

        # elements per spatial direction
        if self.p['dim'] == 2:
            # self.mesh = df.UnitSquareMesh(n, n, self.p.mesh_setting)
            self.mesh = df.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                                 points=((0.0, 0.0), (self.p['length'], self.p['height'])),
                                                 n=(self.p['num_elements_length'], self.p['num_elements_height']),
                                                 cell_type=df.mesh.CellType.quadrilateral)
        elif self.p['dim'] == 3:
            #self.mesh = df.UnitCubeMesh(n, n, n)
            self.mesh = df.mesh.create_box(comm=MPI.COMM_WORLD,
                                           points=[(0.0, 0.0, 0.0), (self.p['length'],
                                                                     self.p['width'],
                                                                     self.p['height'])],
                                           n=[self.p['num_elements_length'],
                                              self.p['num_elements_width'],
                                              self.p['num_elements_height']],
                                           cell_type=df.mesh.CellType.hexahedron)
        else:
            raise ValueError(f'wrong dimension: {self.p["dim"]} is not implemented for problem setup')

    def create_displacement_boundary(self, V):
        # TODO: use Philipps class here, should be done in separate issue
        # define displacement boundary
        # fenics will individually call this function for every node and will note the true or false value.
        def clamped_boundary(x):
            return np.isclose(x[0], 0)

        displacement_bcs = []
        if self.p['dim'] == 2:
            displacement_bcs.append(df.fem.dirichletbc(np.array([0, 0], dtype=ScalarType),
                                    df.fem.locate_dofs_geometrical(V, clamped_boundary),
                                    V))

        elif self.p['dim'] == 3:
            displacement_bcs.append(df.fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType),
                                    df.fem.locate_dofs_geometrical(V, clamped_boundary),
                                    V))

        return displacement_bcs
