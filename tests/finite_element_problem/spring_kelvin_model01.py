from __future__ import annotations

import numpy as np
from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u


class SpringKelvinModel(IncrSmallStrainModel):
    ''' viscoelastic model based on 1D Three Parameter Model with spring and Kelvin body in row

                               |--- E_1: spring ---|
           --- E_0: spring  ---|                   |--
                               |--- eta: damper ---|

    with deviatoric assumptions for 3D generalization (volumetric part of visco strain == 0 damper just working on deviatoric part)
    time integration: backward Euler

    '''
    def __init__(self, parameters: dict[str, float], constraint: Constraint):
        self._constraint = constraint
        self.E0 = parameters["E0"] # elastic modulus
        self.E1 = parameters["E1"] # visco modulus
        self.tau = parameters["tau"] # relaxation time == eta/(2 mu1) for 1D case eta/E1
        if Constraint.UNIAXIAL_STRESS:
            self.nu = 0.0
        else:
            self.nu = parameters["nu"] # Poisson's ratio

        self.I2 = np.zeros(self.stress_strain_dim, dtype=np.float64)  # Identity of rank 2 tensor

        self.factor_E0 = 1.0
        self.factor_E1 = 1.0

        self.compute_elasticity()

    def compute_elasticity(self):
        '''calculates lame constants and elasticity tensor (as self variable) based on constraint type'''

        # lame constants
        self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lam0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu1 = self.E1 / (2.0 * (1.0 + self.nu))

        match self._constraint:
            case Constraint.FULL:
                self.D_0 = np.array(
                    [
                        [2.0 * self.mu0 + self.lam0, self.lam0, self.lam0, 0.0, 0.0, 0.0],
                        [self.lam0, 2.0 * self.mu0 + self.lam0, self.lam0, 0.0, 0.0, 0.0],
                        [self.lam0, self.lam0, 2.0 * self.mu0 + self.lam0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0
                self.I2[2] = 1.0

            case Constraint.PLANE_STRAIN:
                self.D_0 = np.array(
                    [
                        [2.0 * self.mu0 + self.lam0, self.lam0, self.lam0, 0.0],
                        [self.lam0, 2.0 * self.mu0 + self.lam0, self.lam0, 0.0],
                        [self.lam0, self.lam0, 2.0 * self.mu0 + self.lam0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0

            case Constraint.PLANE_STRESS:
                self.D_0 = (
                        self.E0
                        / (1 - self.nu ** 2.0)
                        * np.array(
                    [
                        [1.0, self.nu, 0.0, 0.0],
                        [self.nu, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0 - self.nu)],
                    ]
                )
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0

            case Constraint.UNIAXIAL_STRESS:
                self.D_0 = np.array([[self.E0]])
                self.I2[0] = 1.0
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

        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(-1, self.stress_strain_dim)
        strain_n = history['strain'].reshape(-1, self.stress_strain_dim)
        strain_visco_n = history['strain_visco'].reshape(-1, self.stress_strain_dim)


        if type(self.factor_E0) is float or type(self.factor_E0) is int:

            if del_t == 0:
                # linear step visko strain is zero
                self.D_updated = self.factor_E0 * self.D_0
                mandel_view += strain_increment @ self.D_updated
                tangent[:] = np.tile(self.D_updated.flatten(), n_gauss)

            else:
                # visco step
                factor = (1 / del_t + 1 / self.tau + self.E0 / (self.tau * self.E1))
                deps_visko_list = np.zeros_like(strain_increment)
                for n, eps in enumerate(strain_increment):
                    deps_visko = 1 / factor * (
                            1 / (self.tau * 2 * self.mu1) * mandel_view[n]
                            - 1 / self.tau * strain_visco_n[n]
                            + self.mu0 / (self.tau * self.mu1) * eps
                            + self.lam0 / (self.tau * 2 * self.mu1) * np.sum(eps[:3]) * self.I2
                    )
                    deps_visko_list[n] = deps_visko

                self.D_updated = self.factor_E0 * self.D_0
                mandel_view += strain_increment @ self.D_updated - 2*self.mu0 * deps_visko_list
                self.D_visco  = (1 - self.mu0/(self.tau*self.mu1*factor)) * self.D_updated
                tangent[:] = np.tile(self.D_visco.flatten(), n_gauss)

                strain_visco_n += deps_visko_list
                strain_n += strain_increment


        elif type(self.factor_E0) is np.ndarray:

            self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
            self.lam0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            self.mu1 = self.E1 / (2.0 * (1.0 + self.nu))
            if del_t == 0:
                # linear step visko strain is zero
                mandel_view += (strain_increment.reshape(-1, self.stress_strain_dim) @ self.D_0) * self.factor_E0[:, np.newaxis]
                tangent[:] = np.multiply(np.repeat(self.factor_E0, len(self.D_0.flatten())), np.tile(self.D_0.flatten(), n_gauss))

            else:
                # visco step
                factor = (1 / del_t + 1 / self.tau + self.E0 / (self.tau * self.E1))
                deps_visko_list = np.zeros_like(strain_increment)
                for n, eps in enumerate(strain_increment):
                    deps_visko = 1 / factor[n] * (
                            1 / (self.tau[n] * 2 * self.mu1[n]) * mandel_view[n]
                            - 1 / self.tau[n] * strain_visco_n[n]
                            + self.mu0[n] / (self.tau[n] * self.mu1[n]) * eps
                            + self.lam0[n] / (self.tau[n] * 2 * self.mu1[n]) * np.sum(eps[:3]) * self.I2
                    )
                    deps_visko_list[n] = deps_visko

                mandel_view += (strain_increment @ self.D_0) * self.factor_E0[:, np.newaxis] - 2 * deps_visko_list * self.mu0[:,np.newaxis]
                t_correction = (1 - self.mu0/(self.tau*self.mu1*factor)) * self.factor_E0
                tangent[:] = np.multiply(np.repeat(t_correction, len(self.D_0.flatten())),
                                         np.tile(self.D_0.flatten(), n_gauss))

                strain_visco_n += deps_visko_list
                strain_n += strain_increment

        else:
            raise ValueError("factor must be a float, int, or np.ndarray")





    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return {'strain_visco': self.stress_strain_dim, 'strain': self.stress_strain_dim}

    def update(self) -> None:
        pass

