from __future__ import annotations

import numpy as np
from numba import jit # for performance optimization

from fenics_constitutive import StressStrainConstraint, IncrSmallStrainModel, strain_from_grad_u


# adapted from fenics-constitutive for changable Young's modulus during am simulation via factor (float or np.ndarray)
class LinearElasticityModel(IncrSmallStrainModel):
    def __init__(self, parameters: dict[str, float], constraint: StressStrainConstraint):
        self._constraint = constraint
        E = parameters["E"]
        nu = parameters["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.factor = 1.0
        match constraint:
            case StressStrainConstraint.FULL:
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = np.array(
                    [
                        [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
                        [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
                        [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu],
                    ]
                )
            case StressStrainConstraint.PLANE_STRAIN:
                # We assert that the strain is being provided with 0 in the z-direction
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = np.array(
                    [
                        [2.0 * mu + lam, lam, lam, 0.0],
                        [lam, 2.0 * mu + lam, lam, 0.0],
                        [lam, lam, 2.0 * mu + lam, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu],
                    ]
                )
            case StressStrainConstraint.PLANE_STRESS:
                # We do not make any assumptions about strain in the z-direction
                # This matrix just multiplies the z component by 0.0 which results
                # in a plane stress state
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = (
                    E
                    / (1 - nu**2.0)
                    * np.array(
                        [
                            [1.0, nu, 0.0, 0.0],
                            [nu, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, (1.0 - nu)],
                        ]
                    )
                )
            case StressStrainConstraint.UNIAXIAL_STRAIN:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
                self.D = np.array([[C]])
            case StressStrainConstraint.UNIAXIAL_STRESS:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                self.D = np.array([[E]])
            case _:
                msg = "Constraint not implemented"
                raise NotImplementedError(msg)

    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        assert (
            grad_del_u.size // (self.geometric_dim**2)
            == mandel_stress.size // self.stress_strain_dim
            == tangent.size // (self.stress_strain_dim**2)
        )
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)

        if type(self.factor) is float or type(self.factor) is int:
            self.D_updated = self.factor * self.D
            mandel_view += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D_updated
            tangent[:] = np.tile(self.D_updated.flatten(), n_gauss)

        elif type(self.factor) is np.ndarray:
            mandel_view += (strain_increment.reshape(-1, self.stress_strain_dim) @ self.D) * self.factor[:, np.newaxis]
            tangent[:] = np.multiply(np.repeat(self.factor, len(self.D.flatten())), np.tile(self.D.flatten(), n_gauss))
        else:
            raise ValueError("factor must be a float, int, or np.ndarray")

    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return None

    def update(self) -> None:
        pass




# adapted from fenics-constitutive for changable material parameters during am simulation via array
class VonMises3D(IncrSmallStrainModel):

    """Von Mises Plasticity model with linear isotropic hardening.
    Computation of trial stress state is entirely deviatoric. Volumetric part is added later
    when the stress increment for the current time step is calculated"""

    def __init__(self, param: dict[str, float], constraint: StressStrainConstraint):
        
        self.p_ka = param["p_ka"]  # kappa - bulk modulus
        self.p_mu = param["p_mu"]  # mu - shear modulus
        self.p_y0 = param["p_y0"]  # y0 - initial yield stress
        self.p_y00 = param["p_y00"]  # y00 - final yield stress
        self.p_w = param["p_w"]  # w - saturation parameter

    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(
            -1, self.stress_strain_dim
        )
        # change for material parameters not constant over domain
        if not isinstance(self.p_ka, np.ndarray):
            self.p_ka = np.full(len(strain_increment), self.p_ka)
        if not isinstance(self.p_mu, np.ndarray):
            self.p_mu = np.full(len(strain_increment), self.p_mu)
        if not isinstance(self.p_y0, np.ndarray):
            self.p_y0 = np.full(len(strain_increment), self.p_y0)
        if not isinstance(self.p_y00, np.ndarray):
            self.p_y00 = np.full(len(strain_increment), self.p_y00)
        if not isinstance(self.p_w, np.ndarray):
            self.p_w = np.full(len(strain_increment), self.p_w)

        evaluate_von_mises_3D(
            del_t,
            strain_increment,
            mandel_stress,
            tangent,
            history["alpha"],
            history["eps_n"],
            self.stress_strain_dim,
            self.p_ka,
            self.p_mu,
            self.p_y0,
            self.p_y00,
            self.p_w,
        )

    #def update(self) -> None:
    #    pass

    @property
    def constraint(self) -> StressStrainConstraint:
        return StressStrainConstraint.FULL

    @property
    def history_dim(self) -> dict[str, int]:
        return {"eps_n": self.constraint.stress_strain_dim, "alpha": 1}


@jit
def evaluate_von_mises_3D(
    del_t: float,
    strain_increment: np.ndarray,
    mandel_stress: np.ndarray,
    tangent: np.ndarray,
    alpha: np.ndarray,
    eps_n: np.ndarray,
    stress_strain_dim: int,
    p_ka: float,
    p_mu: float,
    p_y0: float,
    p_y00: float,
    p_w: float,
) -> None:
    xioi = np.array(
        [
            [1, 1, 1, 0, 0, 0],  # 1dyadic1 tensor
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    I2 = np.zeros(stress_strain_dim, dtype=np.float64)  # Identity of rank 2 tensor
    I2[0] = 1.0
    I2[1] = 1.0
    I2[2] = 1.0
    I4 = np.eye(stress_strain_dim, dtype=np.float64)  # Identity of rank 4 tensor
    xpp = I4 - (1 / 3) * xioi  # Projection tensor of rank 4

    stress_view = mandel_stress.reshape(-1, stress_strain_dim)
    tangent_view = tangent.reshape(-1, stress_strain_dim**2)

    eps_n = eps_n.reshape(-1, stress_strain_dim)

    for n, eps in enumerate(strain_increment):
        tr_eps = np.sum(eps[:3])  # trace of strain
        eps_dev = eps - tr_eps * I2 / 3  # deviatoric strain

        # deviatoric trial stress
        del_sigtr = 2 * p_mu[n] * eps_dev  # incremental trial stress
        stress_n_dev = (
            stress_view[n] - np.sum(stress_view[n][:3]) * I2 / 3
        )  # total deviatoric stress in last time step
        sigtr = (
            stress_n_dev + del_sigtr
        )  # total (deviatoric) trial stress in current time step

        # norm of deviatoric trial internal forces
        sigtrn = np.sqrt(np.dot(sigtr, sigtr))

        # trial yield criterion
        phitr = sigtrn - np.sqrt(2 / 3) * (
            p_y0[n] + (p_y00[n] - p_y0[n]) * (1 - np.exp(-p_w[n] * alpha[n]))
        )

        # CHECK YIELD CRITERION
        # elastic - plastic step

        if phitr > 0:
            # initialization
            gamma = 0.0
            xr = 1
            it = 0.0
            tol = 1e-12
            nmax = 100
            # flow direction
            xn = sigtr / sigtrn

            # UPDATE PLASTIC MULTIPLIER VIA NEWTON-RAPHSON SCHEME
            #alpha_n = alpha[n]

            def f(x, _alpha_n, _sigtrn, _p_mu, _p_y0, _p_y00, _p_w):
                return (
                    _sigtrn
                    - 2 * _p_mu * x
                    - np.sqrt(2 / 3)
                    * (
                        _p_y0
                        + (_p_y00 - _p_y0)
                        * (1 - np.exp(-_p_w * (_alpha_n + np.sqrt(2 / 3) * x)))
                    )
                )

            def df(x, _alpha_n, _sigtrn, _p_mu, _p_y0, _p_y00, _p_w):
                return -2 * _p_mu - (2 / 3) * (_p_y00 - _p_y0) * _p_w * np.exp(
                    -_p_w * (_alpha_n + np.sqrt(2 / 3) * x)
                )

            # gamma = newton(f, gamma, fprime=df, args=(alpha_n, sigtrn,p_mu, p_y0,p_y00,p_w,), tol=tol, maxiter=nmax)
            while np.abs(xr) > tol:
                it = it + 1
                # compute residium
                xr = f(gamma, alpha[n], sigtrn, p_mu[n], p_y0[n], p_y00[n], p_w[n])
                # compute tangent
                xg = df(gamma, alpha[n], sigtrn, p_mu[n], p_y0[n], p_y00[n], p_w[n])
                # update plastic flow
                gamma = gamma - xr / xg
                # exit Newton algorithm for iteration > nmax
                if it > nmax:
                    break
                # end of Newton iterration

            # compute tangent with converged gamma
            xg = df(gamma, alpha[n], sigtrn, p_mu[n], p_y0[n], p_y00[n], p_w[n])

            # algorithmic parameters
            xc1 = -1 / xg
            xc2 = gamma / sigtrn

            # ELASTIC STEP
        else:
            xn = np.zeros(6)
            gamma = 0
            xc1 = 0
            xc2 = 0

        # update eps^p_n+1 and alpha
        eps_n[n] += gamma * xn
        alpha[n] += np.sqrt(2 / 3) * gamma

        # determine incremental elastic-plastic stresses (with volumetric part)
        sh = p_ka[n] * tr_eps * I2 + del_sigtr - 2 * p_mu[n] * gamma * xn
        # update total stresses
        stress_view[n] += sh

        # determine elastic-plastic moduli
        aah = (
            p_ka[n] * xioi
            + 2 * p_mu[n] * (1 - 2 * p_mu[n] * xc2) * xpp
            + 4 * p_mu[n] * p_mu[n] * (xc2 - xc1) * np.outer(xn, xn)
        )
        tangent_view[n] = aah.flatten()