from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
from mpi4py import MPI
import dolfinx as df
import ufl
import numpy as np
from fenicsxconcrete.unit_registry import ureg

class SimpleBeamExperiment(Experiment):
    def __init__(self, parameters = None):

        p = Parameters()
        if not parameters:
            # implement default values only for  parameter == None
            # boundary values...
            p['length'] = 5 * ureg('m')
            p['height'] = 1 * ureg('m')
            p['width'] = 0.8 * ureg('m')
            p['dim'] = 2 * ureg('')

        p['mesh_density'] = 4 * ureg('')  # number of elements in vertical direction, the others are set accordingly
        p['bc_setting'] = 'full' * ureg('')  # default boundary setting

        # updating with provided values
        self.p = p + parameters

        # initiaizing parent with parameters
        self.setup()

        # initialize variable top_displacement
        #self.displ_load = ufl.Constant(0.0)  # applied via fkt: apply_displ_load(...)

    def setup(self):
        # computing the number of elements in each direcction
        n_height = int(self.p.mesh_density)
        n_width = int(n_height/self.p.height*self.p.width)
        n_length = int(n_height/self.p.height*self.p.length)
        if (n_length % 2) != 0: # check for odd number
            n_length += 1 # n_length must be even for loading example

        if self.p.dim == 2:
            self.mesh = df.mesh.create_rectangle(MPI.COMM_WORLD,
                                                 [np.array([0., 0.]), np.array([self.p.length.magnitude, self.p.height.magnitude])],
                                                 [n_length, n_height],
                                                 cell_type=df.mesh.CellType.triangle)
        elif self.p.dim == 3:
            self.mesh = df.mesh.create_box(MPI.COMM_WORLD,
                                           [np.array([0., 0., 0.]),
                                            np.array([self.p.length, self.p.width, self.p.height])],
                                           [n_length, n_width, n_height],
                                           cell_type=df.mesh.CellType.tetrahedron)
        else:
            raise Exception(f'wrong dimension {self.p.dim} for problem setup')


    # def create_temp_bcs(self,V):
    #
    #     # Temperature boundary conditions
    #     T_bc1 = df.Expression('t_boundary', t_boundary=self.p.T_bc1+self.p.zero_C, degree=0)
    #     T_bc2 = df.Expression('t_boundary', t_boundary=self.p.T_bc2+self.p.zero_C, degree=0)
    #     T_bc3 = df.Expression('t_boundary', t_boundary=self.p.T_bc3+self.p.zero_C, degree=0)
    #
    #     temp_bcs = []
    #
    #     temp_bcs.append(df.DirichletBC(V, T_bc1, self.boundary_full()))
    #
    #     return temp_bcs

    # def create_displ_bcs(self,V):
    #     if self.p.dim == 2:
    #         dir_id = 1
    #         fixed_bc = ufl.Constant((0, 0))
    #     elif self.p.dim == 3:
    #         dir_id = 2
    #         fixed_bc = df.Constant((0, 0, 0))
    #
    #     # define surfaces, full, left, right, bottom, top, none
    #     def left_support(x, on_boundary):
    #         return df.near(x[0], 0) and df.near(x[dir_id], 0)
    #     def right_support(x, on_boundary):
    #         return df.near(x[0], self.p.length) and df.near(x[dir_id], 0)
    #     def center_top(x, on_boundary):
    #         return df.near(x[0], self.p.length/2) and df.near(x[dir_id], self.p.height)
    #
    #
    #     # define displacement boundary
    #     displ_bcs = []
    #
    #     displ_bcs.append(df.DirichletBC(V, fixed_bc, left_support, method='pointwise'))
    #     displ_bcs.append(df.DirichletBC(V.sub(dir_id), df.Constant(0), right_support, method='pointwise'))
    #     if self.p.dim == 3:
    #         displ_bcs.append(df.DirichletBC(V.sub(1), df.Constant(0), right_support, method='pointwise'))
    #
    #     if self.p['bc_setting'] == 'full':  # not the best default or good name, but this will not break existing tests...
    #         displ_bcs.append(df.DirichletBC(V.sub(dir_id), self.displ_load, center_top, method='pointwise'))
    #     elif self.p['bc_setting'] == 'no_external_load':  # this will allow to add other loads without having to change this
    #         pass #
    #
    #     return displ_bcs
    #
    #
    # def apply_displ_load(self, displacement_load):
    #     """Updates the applied displacement load
    #
    #     Parameters
    #     ----------
    #     top_displacement : float
    #         Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
    #     """
    #     # TODO: implement this when the material
    #     pass
    #     self.displ_load.assign(df.Constant(displacement_load))
