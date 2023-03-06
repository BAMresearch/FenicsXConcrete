from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
import dolfinx as df
import numpy as np
import gmsh
import os
import meshio
import ufl
from mpi4py import MPI

from fenicsxconcrete.unit_registry import ureg
from fenicsxconcrete.boundary_conditions.boundary import plane_at, point_at
from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions


def generate_cylinder_mesh(radius, height, mesh_density):
    """Uses gmsh to generate a cylinder mesh for fenics

    Paramters
    ---------
    radius : float
        radius of the cylinder
    height : float
        height of the cylinder
    mesh_density : float
        defines the size of the elements
        defines the minimum number of element edges in the height of the cylinder

    Returns
    -------
    mesh : dolfin.cpp.mesh.Mesh object
        cylinder mesh for dolfin
    """

    # file names and location
    folder_name = "mesh"
    file_name = "cylinder"
    msh_file = "{}/{}.msh".format(folder_name, file_name)
    xdmf_file = "{}/{}.xdmf".format(folder_name, file_name)

    # start gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)  # only print warnings etc
    gmsh.model.add("cylinder_mesh")  # give the model a name

    # generate cylinder geometry with origin in (0,0,0)
    # syntax: add_cylinder(x,y,z,dx,dy,dz,radius,angle in radian)
    membrane = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, height, radius, angle=2 * np.pi
    )
    gmsh.model.occ.synchronize()
    dim = 3
    # only physical groups get exported
    # syntax: add_physical_group(dim , list of 3d objects, tag)
    gmsh.model.addPhysicalGroup(dim, [membrane], 1)

    # meshing
    characteristic_length = height / mesh_density
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length)
    # setting for minimal length, arbitrarily chosen as half the max value
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length / 2)
    gmsh.model.mesh.generate(dim)

    # save it to disk as msh in folder
    if not os.path.exists(folder_name):  # creat mesh folder if it does nto exists
        os.mkdir(folder_name)

    # write file
    gmsh.write(msh_file)
    gmsh.finalize()

    # convert msh to xdmf
    meshio_mesh = meshio.read(msh_file)
    meshio.write(xdmf_file, meshio_mesh)

    # read xdmf as dolfin mesh
    with df.io.XDMFFile(MPI.COMM_WORLD,xdmf_file,"r") as mesh_file:
        mesh = mesh_file.read_mesh(name="Grid")

    return mesh


class ConcreteCylinderExperiment(Experiment):
    """A cylinder mesh for a uni-axial displacement load"""

    def __init__(self, parameters=None):
        """initializes the object

        Standard parameters are set
        setup function called

        Parameters
        ----------
        parameters : dictionary, optional
            dictionary with parameters that can override the default values
        """
        # initialize a set of default parameters
        p = Parameters()
        # boundary setting
        p["bc_setting"] = "free" * ureg("")  # boundary setting, two options available: fixed and free
        # fixed: constrained at top and bottom in transversal to loading
        # free: no confinement perpendicular to loading surface
        # mesh information
        p["dim"] = 3 * ureg("")  # dimension of problem, 2D or 3D
                                 # 2D version of the cylinder is a rectangle with plane strain assumption
        p["mesh_density"] = 4 * ureg("")  # in 3D: number of faces on the side when generating a polyhedral approximation
                                          # in 2D: number of elements in each direction
        p["radius"] = 75 * ureg("mm")  # radius of cylinder to approximate in mm
        p["height"] = 100 * ureg("mm")  # height of cylinder in mm

        p['degree'] = 2 * ureg('')  # polynomial degree

        p.update(parameters)

        super().__init__(p)

        # initialize variable top_displacement
        self.top_displacement = df.fem.Constant(domain=self.mesh, c=0.0)  # applied via fkt: apply_displ_load(...)

    def setup(self):
        """Generates the mesh based on parameters

        This function is called during __init__
        """

        if self.p['dim'] == 2:
            # build a rectangular mesh to approximate a 2D cylinder
            self.mesh = df.mesh.create_rectangle( MPI.COMM_WORLD, 
                                                 [[0.0, 0.0],[self.p['radius']* 2, self.p['height']],],
                                                 [self.p['mesh_density'], self.p['mesh_density']],
                                                 cell_type=df.mesh.CellType.triangle,)
        elif self.p['dim'] == 3:
            # generates a 3D cylinder mesh based on radius and height
            # to reduce approximation errors due to the linear tetrahedron mesh, the mesh radius is iteratively changed
            # until the bottom surface area matches that of a circle with the initially defined radius
            def create_cylinder_mesh(radius, p):
                # generate cylinder mesh using gmsh
                mesh = generate_cylinder_mesh(radius, p['height'], p['mesh_density'])
                facets = df.mesh.locate_entities_boundary(mesh, 2, plane_at(0.0, 2))
                tdim = mesh.topology.dim
                f_v = mesh.topology.connectivity(tdim - 1, 0).array.reshape(-1, 3)
                entities = df.graph.create_adjacencylist(f_v[facets])
                values = np.full(facets.shape[0], 2, dtype=np.int32)

                ft = df.mesh.meshtags_from_entities(mesh, tdim - 1, entities, values)
                ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
                bottom_area = df.fem.assemble_scalar(df.fem.form(1 * ds(2)))

                return bottom_area, mesh

            # create a discretized cylinder mesh with the same cross-sectional area as the round cylinder
            target_area = np.pi * self.p['radius']**2
            effective_radius = self.p['radius']
            mesh_area = 0 
            area_error = 1e-6
            #
            # iteratively improve the radius of the mesh till the bottom area matches the target
            while abs(target_area - mesh_area) > target_area * area_error:
                # generate mesh
                self.p["mesh_radius"] = effective_radius  # not required, but maybe interesting as metadata
                mesh_area, self.mesh = create_cylinder_mesh(effective_radius, self.p)
                # new guess
                effective_radius = np.sqrt(target_area / mesh_area) * effective_radius

        else:
            raise Exception(f"wrong dimension {self.p['dim']} for problem setup")

    def create_displacement_boundary(self, V):
        """Defines the displacement boundary conditions

        Parameters
        ----------
            V : FunctionSpace
                Function space of the structure

        Returns
        -------
            displ_bc : list
                A list of DirichletBC objects, defining the boundary conditions
        """

        # define boundary conditions generator
        bc_generator = BoundaryConditions(self.mesh, V)

        if self.p['bc_setting'] == "fixed":
            if self.p['dim'] == 2:
                bc_generator.add_dirichlet_bc(self.top_displacement, self.boundary_top(), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_top(), 0, "geometrical", 0)
                bc_generator.add_dirichlet_bc(df.fem.Constant(domain=self.mesh, c=(0, 0)), self.boundary_bottom(), "geometrical")
            elif self.p['dim']== 3:

                bc_generator.add_dirichlet_bc(self.top_displacement, self.boundary_top(), 2, "geometrical", 2)
                bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_top(), 0, "geometrical", 0)
                bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_top(), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(df.fem.Constant(domain=self.mesh, c=(0, 0, 0)), self.boundary_bottom(), "geometrical")

        elif self.p['bc_setting'] == "free":
            if self.p['dim'] == 2:
                bc_generator.add_dirichlet_bc(self.top_displacement, self.boundary_top(), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_bottom(), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(np.float64(0.0), point_at((0, 0)), 0, "geometrical", 0)

            elif self.p['dim'] == 3:
                # getting nodes at the bottom of the mesh to apply correct boundary condition to arbitrary cylinder mesh
                mesh_points = self.mesh.geometry.x  # list of all nodal coordinates
                bottom_points = mesh_points[(mesh_points[:, 2] == 0.0)]  # copying the bottom nodes, z coord = 0.0

                # sorting by x coordinate
                x_min_boundary_point = bottom_points[bottom_points[:, 0].argsort(kind="mergesort")][0]
                x_max_boundary_point = bottom_points[bottom_points[:, 0].argsort(kind="mergesort")][-1]
                # sorting by y coordinate
                y_boundary_point = bottom_points[bottom_points[:, 1].argsort(kind="mergesort")][0]
            
                bc_generator.add_dirichlet_bc(self.top_displacement, self.boundary_top(), 2, "geometrical", 2)
                bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_bottom(), 2, "geometrical", 2)
                bc_generator.add_dirichlet_bc(np.float64(0.0), point_at(x_min_boundary_point), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(np.float64(0.0), point_at(x_max_boundary_point), 1, "geometrical", 1)
                bc_generator.add_dirichlet_bc(np.float64(0.0), point_at(y_boundary_point), 0, "geometrical", 0)
            
            else:
                raise Exception(f"dim setting: {self.p['dim']}, not implemented for cylinder bc setup: free")
        else:
            raise Exception(f"Wrong boundary setting: {self.p['bc_setting']}, for cylinder setup")

        return bc_generator.bcs

    def apply_displ_load(self, top_displacement):
        """Updates the applied displacement load

        Parameters
        ----------
        top_displacement : float
            Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
        """

        self.top_displacement.value = top_displacement.magnitude

    def boundary_top(self):
        if self.p['dim'] == 2:
            return plane_at(self.p['height'], 1)
        elif self.p['dim'] == 3:
            return plane_at(self.p['height'], 2)
    
    def boundary_bottom(self):
        if self.p['dim'] == 2:
            return plane_at(0.0, 1)
        elif self.p['dim'] == 3:
            return plane_at(0.0, 2)