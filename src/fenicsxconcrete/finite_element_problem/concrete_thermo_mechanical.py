import dolfinx as df
import numpy as np
import pint
import ufl
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import (
    Parameters,
    QuadratureEvaluator,
    QuadratureRule,
    project,
)
from fenicsxconcrete.unit_registry import ureg


class ConcreteThermoMechanical(MaterialProblem):
    """
    A class for a weakly coupled thermo-mechanical model, where the youngs modulus of the
    concrete depends on the thermal problem.

    Args:
        experiment: The experimental setup.
        parameters: Dictionary with parameters.
        pv_name: Name of the paraview file, if paraview output is generated.
        pv_path: Name of the paraview path, if paraview output is generated.
    """

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        pv_name: str = "pv_output_full",
        pv_path: str | None = None,
    ) -> None:

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()
        # default_p['dummy'] = 'example' * ureg('')  # example default parameter for this class

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(experiment, default_p, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        description = {
            "igc": "Ideal gas constant",
            "rho": "Density of concrete",
            "themal_cond": "Thermal conductivity",
            "vol_heat_cap": "TODO",
            "Q_pot": "potential heat per weight of binder",
            "Q_inf": "potential heat per concrete volume",
            "B1": "TODO",
            "B2": "TODO",
            "eta": "TODO: something about diffusion",
            "alpha_max": "TODO: also possible to approximate based on equation with w/c",
            "E_act": "activation energy per mol",
            "T_ref": "reference temperature",
            "temp_adjust_law": "TODO",
            "degree": "Polynomial degree for the FEM model",
            "E_28": "Youngs Modulus of concrete",
            "nu": "Poissons Ratio",
            "alpha_t": "TODO",
            "alpha_0": "TODO",
            "a_E": "TODO",
            "fc_inf": "TODO",
            "a_fc": "TODO",
            "ft_inf": "TODO",
            "a_ft": "TODO",
            "evolution_ft": "TODO",
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """
        Static method that returns a set of default parameters.

        Returns:
            The default parameters as a dictionary.
        """
        experiment = CantileverBeam(CantileverBeam.default_parameters())
        # Material parameter for concrete model with temperature and hydration
        default_parameters = {
            "igc": 8.3145 * ureg("J/K/mol"),
            "rho": 2350.0 * ureg("kg/m^3"),
            "themal_cond": 2.0 * ureg("W/(m*K)"),
            "vol_heat_cap": 2.4e6 * ureg("J/(m^3 * K)"),
            "Q_pot": 500e3 * ureg("J/kg"),
            "Q_inf": 144000000 * ureg("J/m^3"),
            "B1": 2.916e-4 * ureg("1/s"),
            "B2": 0.0024229 * ureg(""),
            "eta": 5.554 * ureg(""),
            "alpha_max": 0.875 * ureg(""),
            "T_ref": 25.0 * ureg.degC,
            "temp_adjust_law": "exponential" * ureg(""),
            "degree": 2 * ureg(""),
            "E_28": 15 * ureg("MPa"),
            "nu": 0.2 * ureg(""),
            "alpha_t": 0.2 * ureg(""),
            "alpha_0": 0.05 * ureg(""),
            "a_E": 0.6 * ureg(""),
            "fc_inf": 6210000 * ureg(""),
            "a_fc": 1.2 * ureg(""),
            "ft_inf": 467000 * ureg(""),
            "a_ft": 1.0 * ureg(""),
            "evolution_ft": True * ureg(""),
        }
        default_parameters["E_act"] = 5653.0 * default_parameters["igc"] * ureg("J/mol")
        return experiment, default_parameters

    def setup(self):

        # setting up the two nonlinear problems
        self.temperature_problem = ConcreteTemperatureHydrationModel(
            self.experiment.mesh, self.p, pv_name=self.pv_name
        )

        # here I "pass on the parameters from temperature to mechanics problem.."
        self.mechanics_problem = ConcreteMechanicsModel(self.experiment.mesh, self.p, pv_name=self.pv_name)
        # coupling of the output files
        self.mechanics_problem.pv_file = self.temperature_problem.pv_file

        # initialize concrete temperature as given in experimental setup
        self.set_inital_T(self.p["T_0"])

        # setting bcs
        self.mechanics_problem.set_bcs(self.experiment.create_displ_bcs(self.mechanics_problem.V))
        self.temperature_problem.set_bcs(self.experiment.create_temp_bcs(self.temperature_problem.V))

        # setting up the solvers
        self.temperature_solver = df.nls.petsc.NewtonSolver(self.temperature_problem)
        self.temperature_solver.atol = 1e-9
        self.temperature_solver.rtol = 1e-8

        self.mechanics_solver = df.nls.petsc.NewtonSolver(self.mechanics_problem)
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        # if self.wrapper:
        #     self.wrapper.set_geometry(self.mechanics_problem.V, [])

    def solve(self, t=1.0) -> None:
        if self.wrapper:
            self.wrapper.next_state()
        # print('Solving: T') # TODO ouput only a certain log level INFO

        self.temperature_solver.solve(self.temperature_problem.u)

        # set current DOH for computation of Young's modulus
        self.mechanics_problem.q_alpha = self.temperature_problem.q_alpha
        # print('Solving: u') # TODO ouput only a certain log level INFO

        # mechanics paroblem is not required for temperature, could crash in frist time steps but then be useful
        try:
            self.mechanics_solver.solve(self.mechanics_problem.u)
        except:
            print("AAAAAAAAAAHHHHHHHHHH!!!!!")

        # history update
        self.temperature_problem.update_history()

        # save fields to global problem for sensor output
        self.displacement = self.mechanics_problem.u
        self.temperature = self.temperature_problem.T

        self.degree_of_hydration = project(
            self.temperature_problem.q_alpha, self.temperature_problem.visu_space, self.temperature_problem.rule.dx
        )

        self.q_degree_of_hydration = self.temperature_problem.q_alpha
        self.q_yield = self.mechanics_problem.q_yield
        self.stress = self.mechanics_problem.sigma_ufl

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, self.wrapper, t)
        if self.wrapper:
            self.wrapper.write_state()

    def pv_plot(self, t=0):
        # calls paraview output for both problems
        self.temperature_problem.pv_plot(t=t)
        self.mechanics_problem.pv_plot(t=t)

    def set_inital_T(self, T: float):
        self.temperature_problem.set_initial_T(T)

    def set_timestep(self, dt: float):
        self.temperature_problem.set_timestep(dt)

    def get_heat_of_hydration_ftk(self):
        return self.temperature_problem.heat_of_hydration_ftk

    def get_E_alpha_fkt(self):
        return np.vectorize(self.mechanics_problem.E_fkt)

    def get_X_alpha_fkt(self):
        return self.mechanics_problem.general_hydration_fkt


# class ConcreteTempHydrationModel(df.NonlinearProblem):
#     def __init__(self, mesh, p, pv_name='temp_output', **kwargs):
#         df.NonlinearProblem.__init__(self)  # apparently required to initialize things
#         self.p = p

#         if mesh != None:
#             # initialize possible paraview output
#             self.pv_file = df.XDMFFile(pv_name + '.xdmf')
#             self.pv_file.parameters["flush_output"] = True
#             self.pv_file.parameters["functions_share_mesh"] = True
#             # function space for single value per element, required for plot of quadrature space values

#             # initialize timestep, musst be reset using .set_timestep(dt)
#             self.dt = 0
#             self.dt_form = df.Constant(self.dt)

#             if self.p.degree == 1:
#                 self.visu_space = df.FunctionSpace(mesh, "DG", 0)
#             else:
#                 self.visu_space = df.FunctionSpace(mesh, "P", 1)

#             metadata = {"quadrature_degree": self.p.degree, "quadrature_scheme": "default"}
#             dxm = df.dx(metadata=metadata)

#             # solution field
#             self.V = df.FunctionSpace(mesh, 'P', self.p.degree)

#             # generic quadrature function space
#             cell = mesh.ufl_cell()
#             q = "Quadrature"
#             quadrature_element = df.FiniteElement(q, cell, degree=self.p.degree, quad_scheme="default")
#             q_V = df.FunctionSpace(mesh, quadrature_element)

#             # quadrature functions
#             self.q_T = df.Function(q_V, name="temperature")
#             self.q_alpha = df.Function(q_V, name="degree of hydration")
#             self.q_alpha_n = df.Function(q_V, name="degree of hydration last time step")
#             self.q_delta_alpha = df.Function(q_V, name="inrease in degree of hydration")
#             self.q_ddalpha_dT = df.Function(q_V, name="derivative of delta alpha wrt temperature")

#             # empfy list for newton iteration to compute delta alpha using the last value as starting point
#             self.delta_alpha_n_list = np.full(np.shape(self.q_alpha_n.vector().get_local()), 0.2)
#             # empfy list for newton iteration to compute delta alpha using the last value as starting point
#             self.delta_alpha_guess = np.full(np.shape(self.q_alpha_n.vector().get_local()), 0.5)

#             # scalars for the analysis of the heat of hydration
#             self.alpha = 0
#             self.delta_alpha = 0

#             # Define variational problem
#             self.T = df.Function(self.V)  # temperature
#             self.T_n = df.Function(self.V)  # overwritten later...
#             T_ = df.TrialFunction(self.V)  # temperature
#             vT = df.TestFunction(self.V)

#             # normal form
#             R_ufl = df.Constant(self.p.vol_heat_cap) * (self.T) * vT * dxm
#             R_ufl += self.dt_form * df.dot(df.Constant(self.p.themal_cond) * df.grad(self.T), df.grad(vT)) * dxm
#             R_ufl += -  df.Constant(self.p.vol_heat_cap) * self.T_n * vT * dxm
#             # quadrature point part

#             self.R = R_ufl - df.Constant(
#                 self.p.Q_inf) * self.q_delta_alpha * vT * dxm

#             # derivative
#             # normal form
#             dR_ufl = df.derivative(R_ufl, self.T)
#             # quadrature part
#             self.dR = dR_ufl - df.Constant(
#                 self.p.Q_inf) * self.q_ddalpha_dT * T_ * vT * dxm

#             # setup projector to project continuous funtionspace to quadrature
#             self.project_T = LocalProjector(self.T, q_V, dxm)

#             self.assembler = None  # set as default, to check if bc have been added???

#     def delta_alpha_fkt(self, delta_alpha, alpha_n, T):
#         return delta_alpha - self.dt * self.affinity(delta_alpha, alpha_n) * self.temp_adjust(T)

#     def delta_alpha_prime(self, delta_alpha, alpha_n, T):
#         return 1 - self.dt * self.daffinity_ddalpha(delta_alpha, alpha_n) * self.temp_adjust(T)

#     def heat_of_hydration_ftk(self, T, time_list, dt, parameter):

#         def interpolate(x, x_list, y_list):
#             # assuming ordered x list

#             i = 0
#             # check if x is in the dataset
#             if x > x_list[-1]:
#                 print(' * Warning!!!: Extrapolation!!!')
#                 point1 = (x_list[-2], y_list[-2])
#                 point2 = (x_list[-1], y_list[-1])
#             elif x < x_list[0]:
#                 print(' * Warning!!!: Extrapolation!!!')
#                 point1 = (x_list[0], y_list[0])
#                 point2 = (x_list[1], y_list[1])
#             else:
#                 while x_list[i] < x:
#                     i += 1
#                 point1 = (x_list[i - 1], y_list[i - 1])
#                 point2 = (x_list[i], y_list[i])

#             slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
#             x_increment = x - point1[0]
#             y_increment = slope * x_increment
#             y = point1[1] + y_increment

#             return y

#         # get tmax, identify number of time steps, then interpolate data
#         # assuming time list is ordered!!!
#         tmax = time_list[-1]

#         # set paramters
#         self.p.B1 = parameter['B1']
#         self.p.B2 = parameter['B2']
#         self.p.eta = parameter['eta']
#         self.p.alpha_max = parameter['alpha_max']
#         self.p.E_act = parameter['E_act']
#         self.p.T_ref = parameter['T_ref']
#         self.p.Q_pot = parameter['Q_pot']

#         # set time step
#         self.dt = dt

#         t = 0
#         time = [0.0]
#         heat = [0.0]
#         alpha_list = [0.0]
#         alpha = 0
#         delta_alpha = 0.0

#         error_flag = False
#         while t < tmax:
#             # compute delta_alpha
#             try:
#                 delta_alpha = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha, T + self.p.zero_C),
#                                                     fprime=self.delta_alpha_prime, x0=delta_alpha)
#                 if delta_alpha < 0:
#                     raise Exception(
#                         f'Problem with solving for delta alpha. Result is negative for starting delta alpha = {delta_alpha}')
#             except:
#                 delta_alpha = 0.2
#                 try:
#                     delta_alpha = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha, T + self.p.zero_C),
#                                                         fprime=self.delta_alpha_prime, x0=delta_alpha)
#                     if delta_alpha < 0:
#                         raise Exception(
#                             'Problem with solving for delta alpha. Result is negative for starting delta alpha = 0.2')
#                 except:
#                     delta_alpha = 0.5
#                     try:
#                         delta_alpha = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha, T + self.p.zero_C),
#                                                             fprime=self.delta_alpha_prime, x0=delta_alpha)
#                         if delta_alpha < 0:
#                             raise Exception(
#                                 'Problem with solving for delta alpha. Result is negative for starting delta alpha = 0.5')
#                     except:
#                         delta_alpha = 1.0

#                         try:
#                             delta_alpha = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha, T + self.p.zero_C),
#                                                                 fprime=self.delta_alpha_prime, x0=delta_alpha)
#                             if delta_alpha < 0:
#                                 raise Exception('Problem with solving for delta alpha. Result is negative.')
#                         except:
#                             error_flag = True
#                             break

#             # update alpha
#             alpha = delta_alpha + alpha
#             # save heat of hydration
#             alpha_list.append(alpha)
#             heat.append(alpha * self.p.Q_pot)

#             # timeupdate
#             t = t + self.dt
#             time.append(t)

#         # if there was a probem with the computation (bad input values), return zero
#         if error_flag:
#             heat_interpolated = np.zeros_like(time_list)
#             alpha_interpolated = np.zeros_like(time_list)
#         else:
#             # interpolate heat to match time_list
#             heat_interpolated = []
#             alpha_interpolated = []
#             for value in time_list:
#                 heat_interpolated.append(interpolate(value, time, heat))
#                 alpha_interpolated.append(interpolate(value, time, alpha_list))

#         return np.asarray(heat_interpolated) / 1000, np.asarray(alpha_interpolated)

#     def get_affinity(self):
#         alpha_list = []
#         affinity_list = []
#         for val in range(1000):
#             alpha = val / 1000
#             alpha_list.append(alpha)
#             affinity_list.append(self.affinity(alpha, 0))

#         return np.asarray(alpha_list), np.asarray(affinity_list)

#     def evaluate_material(self):
#         # project temperautre onto quadrature spaces
#         self.project_T(self.q_T)

#         # convert quadrature spaces to numpy vector
#         temperature_list = self.q_T.vector().get_local()
#         alpha_n_list = self.q_alpha_n.vector().get_local()

#         # solve for alpha at each quadrature point
#         # here the newton raphson method of the scipy package is used
#         # the zero value of the delta_alpha_fkt is found for each entry in alpha_n_list is found. the corresponding temparature
#         # is given in temperature_list and as starting point the value of last step used from delta_alpha_n
#         try:
#             delta_alpha_list = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha_n_list, temperature_list),
#                                                      fprime=self.delta_alpha_prime, x0=self.delta_alpha_n_list)
#             # I dont trust the algorithim!!! check if only applicable results are obtained
#         except:
#             # AAAAAAHHHH, negative delta alpha!!!!
#             # NO PROBLEM!!!, different starting value!
#             delta_alpha_list = scipy.optimize.newton(self.delta_alpha_fkt, args=(alpha_n_list, temperature_list),
#                                                      fprime=self.delta_alpha_prime, x0=self.delta_alpha_guess)
#             if np.any(delta_alpha_list < 0.0):
#                 print('AAAAAAHHHH, negative delta alpha!!!!')
#                 raise Exception(
#                     'There is a problem with the alpha computation/initial guess, computed delta alpha is negative.')

#         # save the delta alpha for next iteration as starting guess
#         self.delta_alpha_n_list = delta_alpha_list

#         # compute current alpha
#         alpha_list = alpha_n_list + delta_alpha_list
#         # compute derivative of delta alpha with respect to temperature for rhs
#         ddalpha_dT_list = self.dt * self.affinity(alpha_list, alpha_n_list) * self.temp_adjust_tangent(temperature_list)

#         # project lists onto quadrature spaces
#         set_q(self.q_alpha, alpha_list)
#         set_q(self.q_delta_alpha, delta_alpha_list)
#         set_q(self.q_ddalpha_dT, ddalpha_dT_list)

#     def update_history(self):
#         self.T_n.assign(self.T)  # save temparature field
#         self.q_alpha_n.assign(self.q_alpha)  # save alpha field

#     def set_timestep(self, dt):
#         self.dt = dt
#         self.dt_form.assign(df.Constant(self.dt))

#     def set_initial_T(self, T):
#         # set initial temperature, in kelvin
#         T0 = df.Expression('t_zero', t_zero=T + self.p.zero_C, degree=0)
#         self.T_n.interpolate(T0)
#         self.T.interpolate(T0)

#     def set_bcs(self, bcs):
#         # Only now (with the bcs) can we initialize the assembler
#         self.assembler = df.SystemAssembler(self.dR, self.R, bcs)

#     def F(self, b, x):
#         if self.dt <= 0:
#             raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")
#         if not self.assembler:
#             raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
#         self.evaluate_material()
#         self.assembler.assemble(b, x)

#     def J(self, A, x):
#         self.assembler.assemble(A)

#     def pv_plot(self, t=0):
#         # paraview export

#         # temperature plot
#         T_plot = df.project(self.T, self.V)
#         T_plot.rename("Temperature", "test string, what does this do??")  # TODO: what does the second string do?
#         self.pv_file.write(T_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

#         # degree of hydration plot
#         alpha_plot = df.project(self.q_alpha, self.visu_space,
#                                 form_compiler_parameters={'quadrature_degree': self.p.degree})
#         alpha_plot.rename("DOH", "test string, what does this do??")  # TODO: what does the second string do?
#         self.pv_file.write(alpha_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

#     def temp_adjust(self, T):
#         val = 1
#         if self.p.temp_adjust_law == 'exponential':
#             val = np.exp(-self.p.E_act / self.p.igc * (1 / T - 1 / (self.p.T_ref + self.p.zero_C)))
#         elif self.p.temp_adjust_law == 'off':
#             pass
#         else:
#             # TODO throw correct error
#             raise Exception(
#                 f'Warning: Incorrect temp_adjust_law {self.p.temp_adjust_law} given, only "exponential" and "off" implemented')
#         return val

#         # derivative of the temperature adjustment factor with respect to the temperature

#     def temp_adjust_tangent(self, T):
#         val = 0
#         if self.p.temp_adjust_law == 'exponential':
#             val = self.temp_adjust(T) * self.p.E_act / self.p.igc / T ** 2
#         return val

#     # affinity function
#     def affinity(self, delta_alpha, alpha_n):
#         affinity = self.p.B1 * (self.p.B2 / self.p.alpha_max + delta_alpha + alpha_n) * (
#                 self.p.alpha_max - (delta_alpha + alpha_n)) * np.exp(
#             -self.p.eta * (delta_alpha + alpha_n) / self.p.alpha_max)
#         return affinity

#     # derivative of affinity with respect to delta alpha
#     def daffinity_ddalpha(self, delta_alpha, alpha_n):
#         affinity_prime = self.p.B1 * np.exp(-self.p.eta * (delta_alpha + alpha_n) / self.p.alpha_max) * (
#                 (self.p.alpha_max - (delta_alpha + alpha_n)) * (
#                 self.p.B2 / self.p.alpha_max + (delta_alpha + alpha_n)) * (
#                         -self.p.eta / self.p.alpha_max) - self.p.B2 / self.p.alpha_max - 2 * (
#                         delta_alpha + alpha_n) + self.p.alpha_max)
#         return affinity_prime


class ConcreteMechanicsModel(df.fem.petsc.NonlinearProblem):
    """
    Description of a concrete mechanics model

    Args:
        mesh : The mesh.
        parameters : Dictionary of material parameters.
        rule: The quadrature rule.
        pv_name: Name of the output file.

    """

    def __init__(
        self,
        mesh: df.mesh.Mesh,
        parameters: dict[str, int | float | str | bool],
        rule: QuadratureRule,
        pv_name: str = "mechanics_output",
    ):
        self.p_magnitude = parameters
        dim_to_stress_dim = {1: 1, 2: 3, 3: 6}
        self.stress_strain_dim = dim_to_stress_dim[self.p_magnitude["dim"]]
        self.rule = rule
        self.mesh = mesh
        if self.p_magnitude["degree"] == 1:
            self.visu_space = df.fem.FunctionSpace(mesh, ("DG", 0))
            self.visu_space_T = df.fem.TensorFunctionSpace(mesh, ("DG", 0))
        else:
            self.visu_space = df.fem.FunctionSpace(mesh, ("CG", 1))
            self.visu_space_T = df.fem.TensorFunctionSpace(mesh, ("CG", 1))

        self.V = df.fem.VectorFunctionSpace(mesh, ("CG", self.p_magnitude["degree"]))

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(mesh)

        # q_VT = self.rule.create_quadrature_vector_space(mesh, dim=self.stress_strain_dim)

        # quadrature functions
        self.q_E = df.fem.Function(q_V, name="youngs_modulus")
        # TODO: are the following needed as Functions or can they be arrays?
        # self.q_fc = df.fem.Function(q_V, name="compressive_strength")
        # self.q_ft = df.fem.Function(q_V, name="tensile_strength")
        # self.q_yield = df.fem.Function(q_V, name="yield_criterion")

        # self.q_alpha = df.fem.Function(q_V, name="degree_of_hydration")

        # self.q_sigma = df.fem.Function(q_VT, name="stress_tensor")
        self.q_fc = self.rule.create_array(self.mesh, shape=1)
        self.q_ft = self.rule.create_array(self.mesh, shape=1)
        self.q_yield = self.rule.create_array(self.mesh, shape=1)
        self.q_alpha = self.rule.create_array(self.mesh, shape=1)
        self.q_sigma = self.rule.create_array(self.mesh, shape=self.stress_strain_dim)

        # initialize degree of hydration to 1, in case machanics module is run without hydration coupling
        self.q_alpha[:] = 1.0

        # Define variational problem
        self.u = df.fem.Function(self.V, name="Displacements")
        v = ufl.TestFunction(self.V)

        # Elasticity parameters without multiplication with E
        x_mu = 1.0 / (2.0 * (1.0 + self.p_magnitude["nu"]))
        x_lambda = (
            1.0 * self.p_magnitude["nu"] / ((1.0 + self.p_magnitude["nu"]) * (1.0 - 2.0 * self.p_magnitude["nu"]))
        )

        # Stress computation for linear elastic problem without multiplication with E
        def x_sigma(v):
            return 2.0 * x_mu * ufl.sym(ufl.grad(v)) + x_lambda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v))

        # Volume force
        volume_force = [0.0] * self.p_magnitude["dim"]
        volume_force[-1] = -self.p_magnitude["g"] * self.p_magnitute["rho"]
        f_ufl = ufl.Constant(mesh, tuple(volume_force))

        self.sigma_ufl = self.q_E * x_sigma(self.u)

        R_ufl = self.q_E * ufl.inner(x_sigma(self.u), ufl.sym(ufl.grad(v))) * self.rule.dx
        R_ufl += -df.inner(f_ufl, v) * rule.dx  # add volumetric force, aka gravity (in this case)
        # quadrature point part
        self.R = R_ufl

        # derivative
        # normal form
        dR_ufl = df.derivative(R_ufl, self.u)
        # quadrature part
        self.dR = dR_ufl

        self.sigma_evaluator = QuadratureEvaluator(self.sigma_voigt(self.sigma_ufl), mesh, self.rule)

        super().__init__(self.R, self.u, [], self.dR)
        # self.assembler = None  # set as default, to check if bc have been added???

    def sigma_voigt(self, s):
        # 1D option
        if s.ufl_shape == (1, 1):
            stress_vector = ufl.as_vector((s[0, 0]))
        # 2D option
        elif s.ufl_shape == (2, 2):
            stress_vector = ufl.as_vector((s[0, 0], s[1, 1], s[0, 1]))
        # 3D option
        elif s.ufl_shape == (3, 3):
            stress_vector = ufl.as_vector((s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]))
        else:
            raise ("Problem with stress tensor shape for voigt notation")

        return stress_vector

    def E_fkt(self, alpha: float, parameters: dict) -> float:

        if alpha < parameters["alpha_t"]:
            E = (
                parameters["E_inf"]
                * alpha
                / parameters["alpha_t"]
                * ((parameters["alpha_t"] - parameters["alpha_0"]) / (1 - parameters["alpha_0"])) ** parameters["a_E"]
            )
        else:
            E = (
                parameters["E_inf"]
                * ((alpha - parameters["alpha_0"]) / (1 - parameters["alpha_0"])) ** parameters["a_E"]
            )
        return E

    def general_hydration_fkt(self, alpha: np.ndarray, parameters: dict) -> np.ndarray:
        return parameters["X_inf"] * alpha ** parameters["a_X"]

    def principal_stress(self, stresses: np.ndarray) -> np.ndarray:
        # checking type of problem
        n = stresses.shape[1]  # number of stress components in stress vector
        # finding eigenvalues of symmetric stress tensor
        # 1D problem
        if n == 1:
            principal_stresses = stresses
        # 2D problem
        elif n == 3:
            # the following uses
            # lambda**2 - tr(sigma)lambda + det(sigma) = 0, solve for lambda using pq formula
            p = -(stresses[:, 0] + stresses[:, 1])
            q = stresses[:, 0] * stresses[:, 1] - stresses[:, 2] ** 2

            D = p**2 / 4 - q  # help varibale
            assert np.all(D >= -1.0e-15)  # otherwise problem with imaginary numbers
            sqrtD = np.sqrt(D)

            eigenvalues_1 = -p / 2.0 + sqrtD
            eigenvalues_2 = -p / 2.0 - sqrtD

            # strack lists as array
            principal_stresses = np.column_stack((eigenvalues_1, eigenvalues_2))

            # principal_stress = np.array([ev1p,ev2p])
        elif n == 6:
            principal_stresses = np.empty([len(stresses), 3])
            # currently slow solution with loop over all stresses and subsequent numpy function call:
            for i, stress in enumerate(stresses):
                # convert voigt to tensor, (00,11,22,12,02,01)
                stress_tensor = np.array(
                    [
                        [stress[0], stress[5], stress[4]],
                        [stress[5], stress[1], stress[3]],
                        [stress[4], stress[3], stress[2]],
                    ]
                )
                # TODO: remove the sorting
                principal_stress = np.linalg.eigvalsh(stress_tensor)
                # sort principal stress from lagest to smallest!!!
                principal_stresses[i] = np.flip(principal_stress)

        return principal_stresses

    def yield_surface(self, stresses: np.ndarray, ft: np.ndarray, fc: float) -> np.ndarray:
        # function for approximated yield surface
        # first approximation, could be changed if we have numbers/information
        fc2 = fc
        # pass voigt notation and compute the principal stress
        p_stresses = self.principal_stress(stresses)

        # get the principle tensile stresses
        t_stresses = np.where(p_stresses < 0, 0, p_stresses)

        # get dimension of problem, ie. length of list with principal stresses
        n = p_stresses.shape[1]
        # check case
        if n == 1:
            # rankine for the tensile region
            rk_yield_vals = t_stresses[:, 0] - ft[:]

            # invariants for drucker prager yield surface
            I1 = stresses[:, 0]
            I2 = np.zeros_like(I1)
        # 2D problem
        elif n == 2:

            # rankine for the tensile region
            rk_yield_vals = (t_stresses[:, 0] ** 2 + t_stresses[:, 1] ** 2) ** 0.5 - ft[:]

            # invariants for drucker prager yield surface
            I1 = stresses[:, 0] + stresses[:, 1]
            I2 = ((stresses[:, 0] + stresses[:, 1]) ** 2 - ((stresses[:, 0]) ** 2 + (stresses[:, 1]) ** 2)) / 2

        # 3D problem
        elif n == 3:
            # rankine for the tensile region
            rk_yield_vals = (t_stresses[:, 0] ** 2 + t_stresses[:, 1] ** 2 + t_stresses[:, 2] ** 2) ** 0.5 - ft[:]

            # invariants for drucker prager yield surface
            I1 = stresses[:, 0] + stresses[:, 1] + stresses[:, 2]
            I2 = (
                (stresses[:, 0] + stresses[:, 1] + stresses[:, 2]) ** 2
                - ((stresses[:, 0]) ** 2 + (stresses[:, 1]) ** 2 + (stresses[:, 2]) ** 2)
            ) / 2
        else:
            raise ("Problem with input to yield surface, the array with stress values has the wrong size ")

        J2 = 1 / 3 * I1**2 - I2
        beta = (3.0**0.5) * (fc2 - fc) / (2 * fc2 - fc)
        Hp = fc2 * fc / ((3.0**0.5) * (2 * fc2 - fc))

        dp_yield_vals = beta / 3 * I1 + J2**0.5 - Hp

        # TODO: is this "correct", does this make sense? for a compression state, what if rk yield > dp yield???
        yield_vals = np.maximum(rk_yield_vals, dp_yield_vals)

        return yield_vals

    def evaluate_material(self) -> None:
        # convert quadrature spaces to numpy vector
        alpha_array = self.q_alpha.vector.array

        parameters = {}
        parameters["alpha_t"] = self.p["alpha_t"]
        parameters["E_inf"] = self.p["E_28"]
        parameters["alpha_0"] = self.p["alpha_0"]
        parameters["a_E"] = self.p["a_E"]
        # vectorize the function for speed up
        # TODO: remove vectorization. It does nothing for speed-up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_array = E_fkt_vectorized(alpha_array, parameters)

        parameters = {}
        parameters["X_inf"] = self.p["fc_inf"]
        parameters["a_X"] = self.p["a_fc"]

        fc_array = self.general_hydration_fkt(alpha_array, parameters)

        parameters = {}
        parameters["X_inf"] = self.p["ft_inf"]
        parameters["a_X"] = self.p["a_ft"]

        if self.p["evolution_ft"]:
            ft_array = self.general_hydration_fkt(alpha_array, parameters)
        else:
            # no evolution....
            ft_array = np.full_like(alpha_array, self.p["ft_inf"])

        # now do the yield function thing!!!
        # I need stresses!!!
        # get stress values
        self.sigma_evaluator.evaluate(self.q_sigma)

        sigma_array = self.q_sigma.vector.array.reshape(-1, self.stress_strain_dim)

        # # project lists onto quadrature spaces
        self.q_E.vector.array[:] = E_array
        self.q_fc[:] = fc_array
        self.q_ft[:] = ft_array
        self.q_yield[:] = self.yield_surface(sigma_array, ft_array, fc_array)

    # set_timestep does not seem to do anything
    # def set_timestep(self, dt)->None:
    #     self.dt = dt
    #     self.dt_form.assign(ufl.Constant(self.mesh, self.dt))

    def set_bcs(self, bcs: list[df.fem.DirichletBCMetaClass]) -> None:
        # this name is important, since it is predefined in the super() class
        self.bcs = bcs

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. We override it to calculate the values on the quadrature
        functions.

        Args:
           x: The vector containing the latest solution
        """
        self.evaluate_material()
        super().form(x)

    # def pv_plot(self, t=0):
    #     # paraview export

    #     # displacement plot
    #     u_plot = df.project(self.u, self.V)
    #     u_plot.rename("Displacement", "test string, what does this do??")  # TODO: what does the second string do?
    #     self.pv_file.write(u_plot, t, encoding=df.XDMFFile.Encoding.ASCII)

    #     # Elasticity parameters without multiplication with E
    #     x_mu = 1.0 / (2.0 * (1.0 + self.p.nu))
    #     x_lambda = 1.0 * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

    #     def x_sigma(v):
    #         return 2.0 * x_mu * df.sym(df.grad(v)) + x_lambda * df.tr(df.sym(df.grad(v))) * df.Identity(len(v))

    #     sigma_plot = df.project(
    #         self.sigma_ufl, self.visu_space_T, form_compiler_parameters={"quadrature_degree": self.p.degree}
    #     )
    #     E_plot = df.project(self.q_E, self.visu_space, form_compiler_parameters={"quadrature_degree": self.p.degree})
    #     fc_plot = df.project(self.q_fc, self.visu_space, form_compiler_parameters={"quadrature_degree": self.p.degree})
    #     ft_plot = df.project(self.q_ft, self.visu_space, form_compiler_parameters={"quadrature_degree": self.p.degree})
    #     yield_plot = df.project(
    #         self.q_yield, self.visu_space, form_compiler_parameters={"quadrature_degree": self.p.degree}
    #     )
    #     E_plot.rename("Young's Modulus", "test string, what does this do??")  # TODO: what does the second string do?
    #     fc_plot.rename(
    #         "Compressive strength", "test string, what does this do??"
    #     )  # TODO: what does the second string do?
    #     ft_plot.rename("Tensile strength", "test string, what does this do??")  # TODO: what does the second string do?
    #     yield_plot.rename("Yield surface", "test string, what does this do??")  # TODO: what does the second string do?
    #     sigma_plot.rename("Stress", "test string, what does this do??")  # TODO: what does the second string do?

    #     self.pv_file.write(E_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
    #     self.pv_file.write(fc_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
    #     self.pv_file.write(ft_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
    #     self.pv_file.write(yield_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
    #     self.pv_file.write(sigma_plot, t, encoding=df.XDMFFile.Encoding.ASCII)
