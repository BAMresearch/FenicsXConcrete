from typing import Callable

import dolfinx as df
import numpy as np
import pint
import scipy
import ufl
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import Experiment, SimpleCube
from fenicsxconcrete.finite_element_problem import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import LogMixin, Parameters, QuadratureEvaluator, QuadratureRule, project, ureg


class ConcreteThermoMechanical(MaterialProblem, LogMixin):
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
            "B1": "numerical shape parameter for heat release function",
            "B2": "numerical shape parameter for heat release function",
            "eta": "numerical shape parameter for heat release function, maybe something about diffusion",
            "alpha_max": "maximum degree of hydration, must be <= 1",
            "E_act": "activation energy per mol",
            "T_ref": "reference temperature",
            "degree": "Polynomial degree for the FEM model",
            "q_degree": "Polynomial degree for which the quadrature rule integrates correctly",
            "E_28": "Youngs Modulus of concrete",
            "nu": "Poissons Ratio",
            "alpha_t": "value of DOH where linear function changes to exponential, approx 0.1-0.2",
            "alpha_0": "offset for Young's modulus as a function of DOH, to allow for a dormant phase",
            "a_E": "exponential parameter to change the shape of the function E(DOH)",
            "fc_inf": "reference value for the compressive strength, default infinity, otherwise at alpha_tx",
            "a_fc": "exponential parameter to change the shape of the function fc(DOH)",
            "ft_inf": "reference value for the tensile strength, default infinity, otherwise at alpha_tx",
            "a_ft": "exponential parameter to change the shape of the function ft(DOH)",
            "evolution_ft": "flag to turn off the evolution of the tensile strength",
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """
        Static method that returns a set of default parameters.

        Returns:
            The default parameters as a dictionary.
        """
        experiment = SimpleCube(SimpleCube.default_parameters())
        # Material parameter for concrete model with temperature and hydration
        default_parameters = {
            "igc": 8.3145 * ureg("J/K/mol"),
            "rho": 2350.0 * ureg("kg/m^3"),
            "thermal_cond": 2.0 * ureg("W/(m*K)"),
            "vol_heat_cap": 2.4e6 * ureg("J/(m^3 * K)"),
            # "Q_pot": 500e3 * ureg("J/kg"), only needed for postprocessing
            "Q_inf": 144000000 * ureg("J/m^3"),
            "B1": 2.916e-4 * ureg("1/s"),
            "B2": 0.0024229 * ureg("1/s"),
            "eta": 5.554 * ureg(""),
            "alpha_max": 0.875 * ureg(""),
            "alpha_tx": 0.68 * ureg(""),
            "T_ref": ureg.Quantity(25.0, ureg.degC),
            "q_degree": 2 * ureg(""),
            "E_28": 15 * ureg("MPa"),
            "nu": 0.2 * ureg(""),
            "alpha_t": 0.2 * ureg(""),
            "alpha_0": 0.05 * ureg(""),
            "a_E": 0.6 * ureg(""),
            "fc_inf": 6210000 * ureg(""),
            "a_fc": 1.2 * ureg(""),
            "ft_inf": 467000 * ureg(""),
            "a_ft": 1.0 * ureg(""),
            "evolution_ft": "True" * ureg(""),
        }
        default_parameters["E_act"] = 5653.0 * default_parameters["igc"] * ureg("J/mol")
        return experiment, default_parameters

    def compute_residuals(self) -> None:
        pass

    def setup(self) -> None:
        # TODO: the next line only makes a test pass. I don't like it either
        _ = self.p["rho"]
        self.t = 0.0

        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        displacement_space = df.fem.VectorFunctionSpace(self.experiment.mesh, ("P", self.p["degree"]))
        temperature_space = df.fem.FunctionSpace(self.experiment.mesh, ("P", self.p["degree"]))

        self.fields = SolutionFields(
            displacement=df.fem.Function(displacement_space, name="displacement"),
            temperature=df.fem.Function(temperature_space, name="temperature"),
        )

        bcs_temperature = self.experiment.create_temperature_bcs(temperature_space)
        # setting up the two nonlinear problems
        self.temperature_problem = ConcreteTemperatureHydrationModel(
            self.experiment.mesh, self.p, self.rule, self.fields.temperature, bcs_temperature
        )

        # here I "pass on the parameters from temperature to mechanics problem.."
        bcs_mechanical = self.experiment.create_displacement_boundary(displacement_space)
        body_forces = self.experiment.create_body_force(ufl.TestFunction(displacement_space))

        self.mechanics_problem = ConcreteMechanicsModel(
            self.experiment.mesh,
            self.p,
            self.rule,
            self.fields.displacement,
            bcs_mechanical,
            body_forces,
        )

        # set q_fields now that the solvers are initialized
        plot_space_type = ("DG", 0) if self.p["degree"] == 1 else ("CG", self.p["degree"] - 1)
        self.q_fields = QuadratureFields(
            measure=self.rule.dx,
            plot_space_type=plot_space_type,
            stress=self.mechanics_problem.sigma_voigt(self.mechanics_problem.sigma(self.fields.displacement)),
            degree_of_hydration=self.temperature_problem.q_alpha,
            youngs_modulus=self.mechanics_problem.q_E,
            compressive_strength=self.mechanics_problem.q_fc,
            tensile_strength=self.mechanics_problem.q_ft,
            yield_values=self.mechanics_problem.q_yield,
        )

        # setting up the solvers
        self.temperature_solver = df.nls.petsc.NewtonSolver(self.experiment.mesh.comm, self.temperature_problem)
        self.temperature_solver.atol = 1e-8
        self.temperature_solver.rtol = 1e-8
        self.temperature_solver.max_it = 100
        self.temperature_solver.error_on_nonconvergence = False

        self.mechanics_solver = df.nls.petsc.NewtonSolver(self.experiment.mesh.comm, self.mechanics_problem)
        self.mechanics_solver.atol = 1e-8
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.max_it = 5
        self.mechanics_solver.error_on_nonconvergence = False

        self.plot_space = df.fem.FunctionSpace(self.experiment.mesh, self.q_fields.plot_space_type)
        self.plot_space_stress = df.fem.VectorFunctionSpace(
            self.experiment.mesh, self.q_fields.plot_space_type, dim=self.mechanics_problem.stress_strain_dim
        )

        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w") as f:
            f.write_mesh(self.mesh)

    def solve(self) -> None:
        # from dolfinx import log
        # log.set_log_level(log.LogLevel.INFO)
        self.logger.info(f"Starting solve for temperature at time {self.time}")
        n, converged = self.temperature_solver.solve(self.fields.temperature)

        if not converged:
            raise Exception("Temperature solve did not converge")
        else:
            self.logger.info(f"Temperature solve converged in {n} iterations")

        # set current DOH for computation of Young's modulus
        self.mechanics_problem.q_array_alpha[:] = self.temperature_problem.q_alpha.vector.array
        # print('Solving: u') # TODO ouput only a certain log level INFO

        # mechanics paroblem is not required for temperature, could crash in frist time steps but then be useful
        try:
            n, converged = self.mechanics_solver.solve(self.fields.displacement)
            if not converged:
                self.logger.warning("Mechanics solve did not converge")
            else:
                self.logger.info(f"Mechanics solve converged in {n} iterations")
        except RuntimeError as e:
            print(
                f"An error occured during the mechanics solve. This can happen in the first few solves. Error message {e}"
            )

        # history update
        self.temperature_problem.update_history()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

        self.update_time()

    def pv_plot(self) -> None:
        self.logger.info(f"Writing output to {self.pv_output_file}")
        t = self.time
        self._pv_plot_mechanics(t)
        self._pv_plot_temperature(t)

    def _pv_plot_temperature(self, t) -> None:
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.temperature, t)

            alpha_plot = project(self.q_fields.degree_of_hydration, self.plot_space, self.q_fields.measure)
            alpha_plot.name = "alpha"
            f.write_function(alpha_plot, t)

    def _pv_plot_mechanics(self, t) -> None:
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, t)

            sigma_plot = project(self.q_fields.stress, self.plot_space_stress, self.rule.dx)
            E_plot = project(self.q_fields.youngs_modulus, self.plot_space, self.rule.dx)
            fc_plot = project(self.q_fields.compressive_strength, self.plot_space, self.rule.dx)
            ft_plot = project(self.q_fields.tensile_strength, self.plot_space, self.rule.dx)
            yield_plot = project(self.q_fields.yield_values, self.plot_space, self.rule.dx)

            E_plot.name = "Youngs_Modulus"
            fc_plot.name = "Compressive_strength"
            ft_plot.name = "Tensile_strength"
            yield_plot.name = "Yield_surface"
            sigma_plot.name = "Stress"

            f.write_function(sigma_plot, t)
            f.write_function(E_plot, t)
            f.write_function(fc_plot, t)
            f.write_function(ft_plot, t)
            f.write_function(yield_plot, t)

    def set_timestep(self, dt: float) -> None:
        self.temperature_problem.set_timestep(dt)

    def get_heat_of_hydration_ftk(self) -> Callable:
        return self.temperature_problem.heat_of_hydration_ftk

    def get_E_alpha_fkt(self) -> Callable:
        return np.vectorize(self.mechanics_problem.E_fkt)

    def get_X_alpha_fkt(self) -> Callable:
        return self.mechanics_problem.general_hydration_fkt


class ConcreteTemperatureHydrationModel(df.fem.petsc.NonlinearProblem, LogMixin):
    def __init__(
        self,
        mesh: df.mesh.Mesh,
        parameters: dict[str, int | float | str | bool],
        rule: QuadratureRule,
        temperature: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
    ) -> None:
        self.mesh = mesh
        self.p = parameters
        self.rule = rule
        self.T = temperature
        self.bcs = bcs
        # initialize timestep, musst be reset using .set_timestep(dt)
        self.dt = parameters["dt"]
        self.dt_form = df.fem.Constant(self.mesh, self.dt)

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(self.mesh)

        # quadrature functions
        self.q_alpha = df.fem.Function(q_V, name="degree_of_hydration")
        self.q_delta_alpha = df.fem.Function(q_V, name="inrease_in_degree_of_hydration")
        self.q_ddalpha_dT = df.fem.Function(q_V, name="derivative_of_delta_alpha_wrt_temperature")

        # quadrature arrays
        self.q_array_T = self.rule.create_quadrature_array(self.mesh)
        self.q_array_alpha_n = self.rule.create_quadrature_array(self.mesh)
        # empfy list for newton iteration to compute delta alpha using the last value as starting point
        self.q_array_delta_alpha_n = np.full(np.shape(self.q_array_T), 0.2)
        # empfy list for newton iteration to compute delta alpha using the last value as starting point
        self.delta_alpha_guess = [0.5, 1.0]

        # scalars for the analysis of the heat of hydration
        self.alpha = 0
        self.delta_alpha = 0

        # Define variational problem
        self.T_n = df.fem.Function(self.T.function_space)
        T_ = ufl.TrialFunction(self.T.function_space)
        vT = ufl.TestFunction(self.T.function_space)

        # normal form
        R_ufl = self.p["vol_heat_cap"] * self.T * vT * self.rule.dx
        R_ufl += self.dt_form * ufl.dot(self.p["thermal_cond"] * ufl.grad(self.T), ufl.grad(vT)) * self.rule.dx
        R_ufl += -self.p["vol_heat_cap"] * self.T_n * vT * self.rule.dx
        # quadrature point part

        self.R = R_ufl - self.p["Q_inf"] * self.q_delta_alpha * vT * self.rule.dx

        # derivative
        # normal form
        dR_ufl = ufl.derivative(R_ufl, self.T)
        # quadrature part
        self.dR = dR_ufl - self.p["Q_inf"] * self.q_ddalpha_dT * T_ * vT * self.rule.dx

        # setup projector to project continuous funtionspace to quadrature
        self.temperature_evaluator = QuadratureEvaluator(self.T, self.mesh, self.rule)

        self.T_n.x.array[:] = self.p["T_0"]
        self.T.x.array[:] = self.p["T_0"]

        super().__init__(self.R, self.T, self.bcs, self.dR)

    def delta_alpha_fkt(self, delta_alpha: np.ndarray, alpha_n: np.ndarray, T: np.ndarray) -> np.ndarray:
        return delta_alpha - self.dt * self.affinity(delta_alpha, alpha_n) * self.temp_adjust(T)

    def delta_alpha_prime(self, delta_alpha: np.ndarray, alpha_n: np.ndarray, T: np.ndarray) -> np.ndarray:
        return 1 - self.dt * self.daffinity_ddalpha(delta_alpha, alpha_n) * self.temp_adjust(T)

    def heat_of_hydration_ftk(
        self, T: np.ndarray, time_list: list[float], dt: float, parameter: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        def interpolate(x, x_list, y_list):
            # assuming ordered x list

            i = 0
            # check if x is in the dataset
            if x > x_list[-1]:
                print(" * Warning!!!: Extrapolation!!!")
                point1 = (x_list[-2], y_list[-2])
                point2 = (x_list[-1], y_list[-1])
            elif x < x_list[0]:
                print(" * Warning!!!: Extrapolation!!!")
                point1 = (x_list[0], y_list[0])
                point2 = (x_list[1], y_list[1])
            else:
                while x_list[i] < x:
                    i += 1
                point1 = (x_list[i - 1], y_list[i - 1])
                point2 = (x_list[i], y_list[i])

            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            x_increment = x - point1[0]
            y_increment = slope * x_increment
            y = point1[1] + y_increment

            return y

        # get tmax, identify number of time steps, then interpolate data
        # assuming time list is ordered!!!
        tmax = time_list[-1]

        # set paramters
        self.p["B1"] = parameter["B1"]
        self.p["B2"] = parameter["B2"]
        self.p["eta"] = parameter["eta"]
        self.p["alpha_max"] = parameter["alpha_max"]
        self.p["E_act"] = parameter["E_act"]
        self.p["T_ref"] = parameter["T_ref"]
        self.p["Q_pot"] = parameter["Q_pot"]

        # set time step
        self.dt = dt

        t = 0
        time = [0.0]
        heat = [0.0]
        alpha_list = [0.0]
        alpha = 0
        delta_alpha_list = [0.0, 0.2, 0.5, 1.0]
        delta_alpha_opt = -1.0
        error_flag = False

        while t < tmax:
            for delta_alpha in delta_alpha_list:
                delta_alpha_opt = scipy.optimize.newton(
                    self.delta_alpha_fkt,
                    args=(alpha, T),
                    fprime=self.delta_alpha_prime,
                    x0=delta_alpha,
                )
                if delta_alpha_opt >= 0.0:
                    # success
                    break
            if delta_alpha_opt < 0.0:
                error_flag = True

            # update alpha
            alpha = delta_alpha_opt + alpha
            # save heat of hydration
            alpha_list.append(alpha)
            heat.append(alpha * self.p["Q_pot"])

            # timeupdate
            t = t + self.dt
            time.append(t)

        # if there was a probem with the computation (bad input values), return zero
        if error_flag:
            heat_interpolated = np.zeros_like(time_list)
            alpha_interpolated = np.zeros_like(time_list)
        else:
            # interpolate heat to match time_list
            heat_interpolated = []
            alpha_interpolated = []
            for value in time_list:
                heat_interpolated.append(interpolate(value, time, heat))
                alpha_interpolated.append(interpolate(value, time, alpha_list))

        return np.asarray(heat_interpolated) / 1000, np.asarray(alpha_interpolated)

    def get_affinity(self) -> tuple[np.ndarray, np.ndarray]:
        alpha_list = []
        affinity_list = []
        for val in range(1000):
            alpha = val / 1000
            alpha_list.append(alpha)
            affinity_list.append(self.affinity(alpha, 0))

        return np.asarray(alpha_list), np.asarray(affinity_list)

    def evaluate_material(self) -> None:

        self.temperature_evaluator.evaluate(self.q_array_T)

        # solve for alpha at each quadrature point
        # here the newton raphson method of the scipy package is used
        # the zero value of the delta_alpha_fkt is found for each entry in alpha_n_list is found. the corresponding
        # temparature is given in temperature_list and as starting point the value of last step used from delta_alpha_n
        delta_alpha = scipy.optimize.newton(
            self.delta_alpha_fkt,
            args=(self.q_array_alpha_n, self.q_array_T),
            fprime=self.delta_alpha_prime,
            x0=self.q_array_delta_alpha_n,
        )
        if np.any(delta_alpha < 0.0):
            self.logger.info("Newton method failed, trying different starting points")
            for guess in self.delta_alpha_guess:
                delta_alpha = scipy.optimize.newton(
                    self.delta_alpha_fkt,
                    args=(self.q_array_alpha_n, self.q_array_T),
                    fprime=self.delta_alpha_prime,
                    x0=np.full_like(self.q_array_alpha_n, guess),
                )
                if np.any(delta_alpha >= 0.0):
                    break
        if np.any(delta_alpha < 0.0):
            self.logger.error("HydrationModel: Newton Method failed with new starting points. Negative delta alpha.")
            raise Exception(
                "There is a problem with the alpha computation/initial guess, computed delta alpha is negative."
            )

        # save the delta alpha for next iteration as starting guess
        self.q_array_delta_alpha_n = delta_alpha

        # compute current alpha
        self.q_alpha.vector.array[:] = self.q_array_alpha_n + delta_alpha
        # compute derivative of delta alpha with respect to temperature for rhs
        self.q_ddalpha_dT.vector.array[:] = (
            self.dt
            * self.affinity(self.q_alpha.vector.array, self.q_array_alpha_n)
            * self.temp_adjust_tangent(self.q_array_T)
        )

        self.q_delta_alpha.vector.array[:] = delta_alpha

    def update_history(self) -> None:
        self.T_n.x.array[:] = self.T.x.array  # save temparature field
        self.q_array_alpha_n[:] = self.q_alpha.vector.array  # save alpha field

    def set_timestep(self, dt: float) -> None:
        self.dt = dt
        self.dt_form.value = dt

    def form(self, x: PETSc.Vec) -> None:
        if self.dt <= 0:
            raise RuntimeError("You need to `.set_timestep(dt)` larger than zero before the solve!")

        self.evaluate_material()
        super().form(x)

    # needed for evaluation
    def temp_adjust(self, T: np.ndarray) -> np.ndarray:
        return np.exp(-self.p["E_act"] / self.p["igc"] * (1 / T - 1 / (self.p["T_ref"])))

        # derivative of the temperature adjustment factor with respect to the temperature

    def temp_adjust_tangent(self, T: np.ndarray) -> np.ndarray:
        return self.temp_adjust(T) * self.p["E_act"] / self.p["igc"] / T**2

    # affinity function, needed for material_evaluation
    def affinity(self, delta_alpha: np.ndarray, alpha_n: np.ndarray) -> np.ndarray:
        affinity = (
            self.p["B1"]
            * (self.p["B2"] / self.p["alpha_max"] + delta_alpha + alpha_n)
            * (self.p["alpha_max"] - (delta_alpha + alpha_n))
            * np.exp(-self.p["eta"] * (delta_alpha + alpha_n) / self.p["alpha_max"])
        )
        return affinity

    # derivative of affinity with respect to delta alpha, needed for evaluation
    def daffinity_ddalpha(self, delta_alpha: np.ndarray, alpha_n: np.ndarray) -> np.ndarray:
        affinity_prime = (
            self.p["B1"]
            * np.exp(-self.p["eta"] * (delta_alpha + alpha_n) / self.p["alpha_max"])
            * (
                (self.p["alpha_max"] - (delta_alpha + alpha_n))
                * (self.p["B2"] / self.p["alpha_max"] + (delta_alpha + alpha_n))
                * (-self.p["eta"] / self.p["alpha_max"])
                - self.p["B2"] / self.p["alpha_max"]
                - 2 * (delta_alpha + alpha_n)
                + self.p["alpha_max"]
            )
        )
        return affinity_prime


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
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
        body_forces: ufl.form.Form | None,
    ):
        self.p = parameters
        dim_to_stress_dim = {1: 1, 2: 3, 3: 6}
        self.stress_strain_dim = dim_to_stress_dim[self.p["dim"]]
        self.rule = rule
        self.mesh = mesh

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(mesh)

        # quadrature functions
        self.q_E = df.fem.Function(q_V, name="youngs_modulus")

        self.q_fc = df.fem.Function(q_V)
        self.q_ft = df.fem.Function(q_V)
        self.q_yield = df.fem.Function(q_V)
        self.q_array_alpha = self.rule.create_quadrature_array(self.mesh, shape=1)
        self.q_array_sigma = self.rule.create_quadrature_array(self.mesh, shape=self.stress_strain_dim)

        # initialize degree of hydration to 1, in case machanics module is run without hydration coupling
        self.q_array_alpha[:] = 1.0

        # Define variational problem
        v = ufl.TestFunction(u.function_space)

        # Elasticity parameters without multiplication with E
        self.x_mu = 1.0 / (2.0 * (1.0 + self.p["nu"]))
        self.x_lambda = 1.0 * self.p["nu"] / ((1.0 + self.p["nu"]) * (1.0 - 2.0 * self.p["nu"]))

        R_ufl = ufl.inner(self.sigma(u), self.eps(v)) * self.rule.dx
        if body_forces is not None:
            R_ufl += body_forces

        self.R = R_ufl

        # derivative
        # normal form
        dR_ufl = ufl.derivative(R_ufl, u)
        # quadrature part
        self.dR = dR_ufl
        self.sigma_evaluator = QuadratureEvaluator(self.sigma_voigt(self.sigma(u)), self.mesh, self.rule)
        super().__init__(self.R, u, bcs, self.dR)

    def _x_sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        eps = ufl.sym(ufl.grad(v))
        x_sigma = 2.0 * self.x_mu * eps + self.x_lambda * ufl.tr(eps) * ufl.Identity(len(v))
        return x_sigma

    def sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self.q_E * self._x_sigma(v)

    def eps(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return ufl.sym(ufl.grad(v))

    def E_fkt(self, alpha: float, parameters: dict) -> float:
        parameters["E_inf"] = (
            parameters["E"]
            / ((parameters["alpha_tx"] - parameters["alpha_0"]) / (1 - parameters["alpha_0"])) ** parameters["a_E"]
        )

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

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. We override it to calculate the values on the quadrature
        functions.

        Args:
           x: The vector containing the latest solution
        """
        self.evaluate_material()
        super().form(x)

    def sigma_voigt(self, s: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
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

    def evaluate_material(self) -> None:
        # convert quadrature spaces to numpy vector
        # TODO: Why these new parameters?
        parameters = {}
        parameters["alpha_t"] = self.p["alpha_t"]
        parameters["E"] = self.p["E_28"]
        parameters["alpha_0"] = self.p["alpha_0"]
        parameters["a_E"] = self.p["a_E"]
        parameters["alpha_tx"] = self.p["alpha_tx"]
        # vectorize the function for speed up
        # TODO: remove vectorization. It does nothing for speed-up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        self.q_E.vector.array[:] = E_fkt_vectorized(self.q_array_alpha, parameters)
        self.q_E.x.scatter_forward()

        # from here postprocessing
        parameters = {}
        parameters["X_inf"] = self.p["fc_inf"]
        parameters["a_X"] = self.p["a_fc"]

        self.q_fc.vector.array[:] = self.general_hydration_fkt(self.q_array_alpha, parameters)
        self.q_fc.x.scatter_forward()

        parameters = {}
        parameters["X_inf"] = self.p["ft_inf"]
        parameters["a_X"] = self.p["a_ft"]

        if self.p["evolution_ft"] == "True":
            self.q_ft.vector.array[:] = self.general_hydration_fkt(self.q_array_alpha, parameters)
        else:
            # no evolution....
            self.q_ft.vector.array[:] = np.full_like(self.q_array_alpha, self.p["ft_inf"])
        self.q_ft.x.scatter_forward()

        self.sigma_evaluator.evaluate(self.q_array_sigma)
        # print(self.q_E.vector.array.shape, self.q_array_sigma.shape)
        # self.q_array_sigma *= self.q_E.vector.array

        self.q_yield.vector.array[:] = self.yield_surface(
            self.q_array_sigma.reshape(-1, self.stress_strain_dim), self.q_ft.vector.array, self.q_fc.vector.array
        )

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
            principal_stresses = np.zeros([len(stresses), 3])
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
                try:
                    # TODO: remove the sorting
                    principal_stress = np.linalg.eigvalsh(stress_tensor)
                    # sort principal stress from lagest to smallest!!!
                    principal_stresses[i] = np.flip(principal_stress)
                except np.linalg.LinAlgError as e:
                    pass

        return principal_stresses

    def yield_surface(self, stresses: np.ndarray, ft: np.ndarray, fc: float) -> np.ndarray:
        # TODO: it does not make sense anymore to include this postprocessing step in the material class
        #             I would suggest to create a sensor, that reads stress and outputs this yield value
        #             but I would maybe add that to future problems if you agree, otherwise leave it and ignore this

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
