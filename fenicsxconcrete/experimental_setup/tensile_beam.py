from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
import dolfinx as df
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from fenicsxconcrete.unit_registry import ureg
import pint
#from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem

class TensileBeam(Experiment):
    def __init__(self, parameters: dict[str, pint.Quantity]):
        # initialize a set of "basic parameters"
        default_p = Parameters()
        default_p['dummy'] = 'example' * ureg('')  # example default parameter for this class

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(default_p)

    def setup(self):
        """defines the mesh for 2D or 3D"""

        if self.p['dim'] == 2:
            self.mesh = df.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                                 points=[(0.0, 0.0),
                                                         (self.p['length'], self.p['height'])],
                                                 n=(self.p['num_elements_length'],
                                                    self.p['num_elements_height']),
                                                 cell_type=df.mesh.CellType.quadrilateral)
        elif self.p['dim'] == 3:
            self.mesh = df.mesh.create_box(comm=MPI.COMM_WORLD,
                                           points=[(0.0, 0.0, 0.0),
                                                   (self.p['length'], self.p['width'], self.p['height'])],
                                           n=[self.p['num_elements_length'],
                                              self.p['num_elements_width'],
                                              self.p['num_elements_height']],
                                           cell_type=df.mesh.CellType.hexahedron)
        else:
            raise ValueError(f'wrong dimension: {self.p["dim"]} is not implemented for problem setup')

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """returns a dictionary with required parameters and a set of working values as example"""

        setup_parameters = {}
        setup_parameters['length'] = 1 * ureg('m')
        setup_parameters['height'] = 0.3 * ureg('m')
        setup_parameters['width'] = 0.3 * ureg('m')  # only relevant for 3D case
        setup_parameters['dim'] = 3 * ureg('')
        setup_parameters['num_elements_length'] = 10 * ureg('')
        setup_parameters['num_elements_height'] = 3 * ureg('')
        setup_parameters['num_elements_width'] = 3 * ureg('')  # only relevant for 3D case
        setup_parameters['load'] = 2000 * ureg('kN')

        return setup_parameters

    def create_displacement_boundary(self, V) -> list:
        # define displacement boundary

        # fenics will individually call this function for every node and will note the true or false value.
        def clamped_boundary(x):
            return np.isclose(x[0], 0)

        displacement_bcs = []

        zero = np.zeros(self.p['dim'])
        displacement_bcs.append(df.fem.dirichletbc(np.array(zero, dtype=ScalarType),
                                                   df.fem.locate_dofs_geometrical(V, clamped_boundary),
                                                   V))

        return displacement_bcs

    def create_force_boundary(self,v):
        boundaries = [(1, lambda x: np.isclose(x[0], self.p['length'])),
        (2, lambda x: np.isclose(x[0], 0))]

        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = df.mesh.locate_entities(self.mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = df.mesh.meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

        force_vector = np.zeros(self.p['dim'])
        force_vector[0] = self.p['load']

        T = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = ufl.dot(T, v) * _ds(1)

        return L
