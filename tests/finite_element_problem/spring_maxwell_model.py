from __future__ import annotations

import numpy as np
from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u


class SpringMaxwellModel(IncrSmallStrainModel):
    ''' viscoelastic model based on 1D Three Parameter Model with spring and Maxwell body in parallel

             |----------- E_0: spring  ----------|
           --|                                   |--
             |--- E_1: spring --- eta: damper ---|

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


    def compute_elasticity(self,E0:float,E1:float,nu:float):
        '''calculates lame constants and elasticity tensor based on constraint type
        Args:
            E0,E1,nu: material parameters

        Returns:
            mu0,mu1,lam0: lame constants
        '''
        # lame constants (need to be updated if time dependent material parameters are used)
        mu0 = E0 / (2.0 * (1.0 + nu))
        lam0 = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu1 = E1 / (2.0 * (1.0 + nu))
        lam1 = E1 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        match self._constraint:
            case Constraint.FULL:
                self.D_0 = np.array(
                    [
                        [2.0 * mu0 + lam0, lam0, lam0, 0.0, 0.0, 0.0],
                        [lam0, 2.0 * mu0 + lam0, lam0, 0.0, 0.0, 0.0],
                        [lam0, lam0, 2.0 * mu0 + lam0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * mu0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu0],
                    ]
                )
                self.D_1 = np.array(
                    [
                        [2.0 * mu1 + lam1, lam1, lam1, 0.0, 0.0, 0.0],
                        [lam1, 2.0 * mu1 + lam1, lam1, 0.0, 0.0, 0.0],
                        [lam1, lam1, 2.0 * mu0 + lam1, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu1, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * mu1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu1],
                    ]
                )

            case Constraint.PLANE_STRAIN:
                self.D_0 = np.array(
                    [
                        [2.0 * mu0 + lam0, lam0, lam0, 0.0],
                        [lam0, 2.0 * mu0 + lam0, lam0, 0.0],
                        [lam0, lam0, 2.0 * mu0 + lam0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu0],
                    ]
                )
                self.D_1 = np.array(
                    [
                        [2.0 * mu1 + lam1, lam1, lam1, 0.0],
                        [lam1, 2.0 * mu1 + lam1, lam1, 0.0],
                        [lam1, lam1, 2.0 * mu1 + lam1, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu1],
                    ]
                )

            case Constraint.PLANE_STRESS:
                self.D_0 = (
                        E0
                        / (1 - nu ** 2.0)
                        * np.array(
                    [
                        [1.0, nu, 0.0, 0.0],
                        [nu, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0 - nu)],
                    ]
                )
                )
                self.D_1 = (
                        E1
                        / (1 - nu ** 2.0)
                        * np.array(
                    [
                        [1.0, nu, 0.0, 0.0],
                        [nu, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0 - nu)],
                    ]
                )
                )

            case Constraint.UNIAXIAL_STRESS:
                self.D_0 = np.array([[E0]])
                self.D_1 = np.array([[E1]])
            case _:
                msg = "Constraint not implemented"
                raise NotImplementedError(msg)

        return mu0, mu1, lam0

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

        # check type of material parameters
        if type(self.E0) is np.ndarray:
            # material parameters gausspoint vise
            update = True
        else:
            # constant material parameters
            E0 = self.E0
            E1 = self.E1
            tau = self.tau
            nu = self.nu
            mu0, mu1, lam0 = self.compute_elasticity(E0, E1, nu)

        # reshape gauss point arrays
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim ** 2)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(-1, self.stress_strain_dim)
        strain_visco_n = history['strain_visco'].reshape(-1, self.stress_strain_dim)
        strain_n = history['strain'].reshape(-1, self.stress_strain_dim)

        # loop over gauss points
        for n, eps in enumerate(strain_increment):

            if update:
                E0 = self.E0[n]
                E1 = self.E1[n]
                tau = self.tau[n]
                nu = self.nu
                mu0, mu1, lam0 = self.compute_elasticity(E0, E1, nu)

            if del_t == 0:
                # linear step visko strain is zero
                dstress = self.D_0 @ eps + self.D_1 @ eps
                D = self.D_0 + self.D_1

            else:
                strain_total = strain_n[n] + eps
                factor = (1 / del_t + 1 / tau)
                deps_visko = 1/factor * (
                              1 / (tau * 2 * mu1) * self.D_1 @ strain_total
                              - 1 / tau * strain_visco_n[n]
                              )

                dstress = self.D_0 @ eps + self.D_1 @ eps - 2*mu1 * deps_visko
                D = self.D_0 + (1 - 1/(tau*factor)) * self.D_1

                # update values
                strain_visco_n[n] += deps_visko

            mandel_view[n] += dstress
            strain_n[n] += eps
            tangent_view[n] = D.flatten()



    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return {'strain_visco': self.stress_strain_dim, 'strain': self.stress_strain_dim}

    def update(self) -> None:
        pass

