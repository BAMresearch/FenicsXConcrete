import logging

import dolfinx
import numpy as np
import scipy as sc
import ufl


class Randomfield(object):
    def __init__(
        self, fct_space, cov_name="squared_exp", mean=0, rho1=1, rho2=1, sigma=1, k=None, ktol=None, _type=""
    ):
        """
        Class for random field
        generates a random field using Karhunen Loeve decomposition
        field of form: mean + sum_i sqr(lambda_i) EV_i xi_i
                lambda_i, EV_i: eigenvalues and -vectors of covariance matrix C=c(r)
                xi_i: new variables of random field (in general gaussian distributed N(0,1**2)
                c(r=|x-y|): covariance function for distance between to space nodes x and y


        Attributes:
            logger: logger object
            cov_name: name of covariance function to use
            mean: mean of random field
            rho1: correlation length x
            rho2: correlation length z
            sigma2: sigma**2 in covariance function
            V: FE fct space for problem
            k: number of eigenvectors to compute
            ktol: tolerance to chose k (norm eigenvalue_k < ktol)
            C: correlation matrix
            M: mass matrix for eigenvalue problem
            lambdas: eigenvalues
            EV: eigenvectors
            field: a representation of the field
            values: values of the variables (set by user or choosen randomly)
            values_means: values of the variables for asymptotic expansion to be set by user
            _type: special case for correlation matrix

        Args:
            fct_space: FE function space
            cov_name: string with name of covariance function (e.g. exp, squared_exp ..)
            mean: mean value of random field
            rho1: correlation length x
            rho2: correlation length z
            sigma: covariance standard deviation
            k: number of eigenvectors to compute
            ktol: tolerance to chose k (norm eigenvalue_k < ktol)
            _type: special cases
        """

        self.logger = logging.getLogger(__name__)

        self.cov_name = cov_name  # name of covariance function to use
        self.cov = getattr(self, "cov_" + cov_name)  # select the right function
        self.mean = mean  # mean of random field
        # self.rho = rho  # correlation length
        self.rho1 = rho1  # correlation length x
        self.rho2 = rho2  # correlation length z
        self.sigma2 = sigma**2  # sigma**2 in covariance function
        self.V = fct_space  # FE fct space for problem
        if k:
            self.k = k
        self.ktol = ktol

        self.C = None  # correlation matrix
        self.M = None  # mass matrix for eigenvalue problem
        self.lambdas = []  # eigenvalues
        self.EV = []  # eigenvectors
        self.k = k  # number of eigenvalues
        self._type = _type  # specail case for correlation matrix

        self.field = None  # a representation of the field
        self.values = np.zeros(self.k)  # values of the variables (set by user or choosen randomly)

        self.values_means = np.zeros(self.k)  # values of the variables for asymptotic expansion to be set by user!!

    def __str__(self):
        """
        Overloaded description of the field.
        """
        name = self.__class__.__name__
        name += "random field with cov fct %s, mean %s, rho1 %s, rho2 %s, k %s, sig2 %s"
        return name % (self.cov_name, self.mean, self.rho1, self.rho2, self.k, self.sigma2)

    def __repr__(self):
        """
        Overloaded description of the field.
        """

        return str(self)

    def cov_exp(self, r):
        """
        Exponential covariance function: sig^2 exp(-r/rho)

        Args:
            r: correlation lenghth

        Returns:
            The covarinace sig^2 exp(-r/rho)
        """
        # TODO: This does not work if rho
        return self.sigma2 * np.exp(-1 / self.rho * r)

    def cov_squared_exp(self, r1, r2):
        """
        2D-Exponential covariance function

        Args:
            r1: correlation length in x
            r2: correlation length in y

        Returns:
            The covariance Squared Exponential covariance function
        """
        return self.sigma2 * np.exp((-1 / (2 * self.rho1**2) * r1**2) - (1 / (2 * self.rho2**2) * r2**2))

    def generate_C(self):
        """
        Generate the covariance matrix for the random field representation
        based on tutorial at http://www.wias-berlin.de/people/marschall/lesson3.html

        Returns:
            self
        """

        # directly use dof coordinates (dof_to_vertex map does not work in higher order spaces)
        coords = self.V.tabulate_dof_coordinates()

        self.logger.debug("shape coordinates %s", coords.shape)

        # evaluate covariance matrix
        L = coords.shape[0]
        c0 = np.repeat(coords, L, axis=0)
        c1 = np.tile(coords, [L, 1])
        # TODO: This does not work if "x"
        if self._type == "x":
            r = np.absolute(c0[:, 0] - c1[:, 0])
        else:
            r1 = c0[:, 0] - c1[:, 0]
            r2 = c0[:, 1] - c1[:, 1]
        C = self.cov(r1, r2)

        C.shape = [L, L]

        self.C = np.copy(C)

        return self

    def solve_covariance_EVP(self):
        """
        Solve generalized eigenvalue problem to generate decomposition of C
        based on tutorial at http://www.wias-berlin.de/people/marschall/lesson3.html

        Returns:
            self
        """

        def get_A(A, B):
            return np.dot(A, np.dot(B, A))

        self.generate_C()

        # mass matrix
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # assemble mass matrix and convert to scipy
        aa = dolfinx.fem.form(ufl.dot(u, v) * ufl.dx)
        M = dolfinx.fem.assemble_matrix(aa)  # dolfin.assemble(u * v * ufl.dx)
        self.M = M
        self.M = M.to_dense()

        self.logger.debug("shape of M %s", self.M.shape)

        # solve generalized eigenvalue problem
        A = get_A(self.M, self.C)

        self.logger.debug("shape of A %s", A.shape)
        w, v = sc.linalg.eigh(A, self.M)

        self.logger.info("EVP size: %s, %s, %s", A.shape, w.shape, v.shape)

        # start with largest eigenvalue
        w_reverse = w[::-1]
        v_reverse = np.flip(v, 1)

        # compute self.k if self.ktol is given
        if self.ktol != None:
            self.logger.debug("self.ktol %s", self.ktol)
            normed_lambdas = w_reverse / w_reverse[0]
            self.logger.debug("normed_lambdas %s", normed_lambdas)
            self.k = np.argmax(normed_lambdas <= self.ktol) + 1  # EV with smaller ktol will not be used
            self.logger.info("required number of modes is %s (according tolerance %s)", self.k, self.ktol)
            if self.k == 0:
                raise ValueError('cannot select enough modes - tolerance "ktol" to small')

        # selected vectors / all values for plotting afterwards
        self.lambdas = w_reverse
        self.EV = v_reverse[:, 0 : self.k]

        self.logger.debug("eigenvalues %s", self.lambdas)

        return self

    def create_random_field(self, _type="random", _dist="N", plot=False, evp="generalized"):
        """
        Create a fixed random field using random or fixed values for the new variables

        Args:
            _type: how to choose the values for the random field variables (random or given)
            _dist: which type of random field if 'N' standard gaussian if 'LN' lognormal field computed as exp(gauss field)
            plot: if True plot eigenvalue decay

        Returns
            Self
        """

        # generate decomposition:

        if len(self.lambdas) <= self.k:
            if evp == "generalized":
                self.solve_covariance_EVP()  # generalized eigenvalue problem
            elif evp == "standard":
                self.solve_covariance_EVP_02()  # standard eigenvalue problem
            else:
                raise ValueError(f"Eigenvalue problems can only be in standard or generalized form. Introduced {evp}.")
        # else already computed

        if plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.title("Eigenvalue decay of covariance matrix")
            plt.semilogy(np.arange(len(self.lambdas)) + 1, self.lambdas)
            plt.axvline(x=self.k - 1)
            plt.show()

        # generate representations of variables
        # create the output field gaussian random field
        self.field = dolfinx.fem.Function(self.V)
        new_data = np.zeros(len(self.field.vector[:]))
        new_data += np.copy(self.mean)

        if _type == "random":
            # random selection of gaussian variables
            self.values = np.random.normal(0, 1, self.k)
            self.logger.info("choose random values for xi %s", self.values)
        elif _type == "given":
            self.logger.info("choose given values for xi %s", self.values)

        # compute field representation with number of modes given in self.k or by self.ktol
        self.lambdas = self.lambdas[0 : self.k]
        self.EV = self.EV[:, 0 : self.k]
        new = np.dot(self.EV, np.diag(np.sqrt(self.lambdas)))
        new_data += np.dot(new, self.values)

        if _dist == "N":
            self.logger.info("N given -> computes standard gauss field")
            self.field.vector[:] = new_data[:]
        elif _dist == "LN":
            self.logger.info("LN given -> computes exp(gauss field)")
            self.field.vector[:] = np.exp(new_data[:])
        else:
            self.logger.error("distribution type <%s> not implemented", _dist)
            raise ValueError("distribution type not implemented")

        return self

    def modes(self, num, plot=False):
        """
        Create fenics functions for the first 'num' modes of the random field (= sqrt(\lambda)*eigenvector)

        Args:
            num: specify number of functions needed
            plot: plots the eigenvalue decay and modes if true
        Returns:
            out: Output field
        """

        # output field
        out = list()
        # check if eigenvectors already computed
        if len(self.lambdas) < num:
            self.solve_covariance_EVP()  # generalized eigenvalue problem

        # transform discrete EV to fenics functions
        for i in range(num):
            fct = dolfinx.fem.Function(self.V)
            fct.vector[:] = self.EV[:, i] * np.sqrt(self.lambdas[i])
            out.append(fct)

        self.logger.info(
            f"eigenvalue[{self.k}-1]/eigenvalue[0]: {self.lambdas[self.k-1]/self.lambdas[0]}; eigenvalue[{self.k}-1] = {self.lambdas[self.k-1]} "
        )

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.title("Eigenvalue decay of covariance matrix")
            plt.semilogy(np.arange(len(self.lambdas)) + 1, 1 / self.lambdas[0] * self.lambdas, "-*r", label="normed")
            plt.axvline(x=num)  # start with 1!!
            # plt.show()
            if self.V.mesh().topology().dim() == 1:
                plt.figure(2)
                plt.title("Eigenvectors \Phi_i \sqrt(\lambda_i)")
                for i in range(num):
                    plt.plot(
                        out[i].function_space().tabulate_dof_coordinates()[:],
                        out[i].vector()[:],
                        "*",
                        label="EV %s" % (i),
                    )  # plotting over dof coordinates!! only as points because order not ongoing
                    plt.legend()
            else:
                for i in range(num):
                    plt.figure("10" + str(i))
                    dolfinx.plot(self.V.mesh())
                    plt_mode = dolfinx.plot(out[i])
                    plt.colorbar(plt_mode)
                    plt.title("Eigen mode scaled with \sqrt(\lambda_i) %s" % (i))

            plt.show()

        self.mode_data = out

        return out

    def save_modes_txt(self, file):
        """
        Save modes in txt file for 1D problems

        Args:
            file: filedirectory name

        Returns:
            True if success
        """
        self.logger.info("saving modes in txt file %s", file)
        try:
            a = len(self.mode_data)
        except:
            # generate modes
            out = self.modes(self.k)
            a = len(self.mode_data)

        if self.V.mesh().topology().dim() == 1:
            x = self.V.tabulate_dof_coordinates()[:]
            data_out = np.zeros((len(x), a))
            for i in range(a):
                data_out[:, i] = self.mode_data[i].vector()[:]

            np.savetxt(file + ".txt", np.c_[x, data_out])

        if self.V.mesh().topology().dim() > 1:
            # save as pxdmf file
            ##file_pvd=dolfin.File(file+'.pvd')
            # file_xdmf = dolfin.XDMFFile(file+'.xdmf')
            for i in range(a):
                self.mode_data[i].rename("E_i", "E_i")
                ##file_pvd << self.mode_data[i],i
                # file_xdmf.write(self.mode_data[i],i)

        return True

    def solve_covariance_EVP_02(self):
        """
        Solve eigenvalue problem assuming massmatrix == I --> standard eigenvalue problem
        to generate decomposition of C
        Returns:
            self
        """

        self.generate_C()

        # solve generalized eigenvalue problem
        A = self.C
        w, v = np.linalg.eigh(A)  # solve standard eigenvalue problem (faster) Eigenvalues in increasing order!!

        self.logger.info("EVP size: %s, %s, %s", A.shape, w.shape, v.shape)

        # start with largest eigenvalue
        w_reverse = w[::-1]
        v_reverse = np.flip(v, 1)

        # compute self.k if self.ktol is given
        if self.ktol != None:
            normed_lambdas = w_reverse / w_reverse[0]
            self.logger.debug("normed_lambdas %s", normed_lambdas)
            self.k = np.argmax(normed_lambdas <= self.ktol) + 1  # index starts with 0
            self.logger.info("required number of modes is %s (according tolerance %s)", self.k, self.ktol)

        # selected vectors / all values for plotting afterwards
        self.lambdas = w_reverse
        self.EV = v_reverse[:, 0 : self.k]

        self.logger.info("eigenvalues %s", self.lambdas)
        return self
