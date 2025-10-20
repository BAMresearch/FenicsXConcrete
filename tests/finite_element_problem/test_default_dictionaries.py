import copy

import pytest

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.finite_element_problem.fenics_constitutive import FenicsConstitutive
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.finite_element_problem.concrete_am_fc import ConcreteAMFC
#from fenicsxconcrete.finite_element_problem.concrete_thermo_mechanical import ConcreteThermoMechanical
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("material_model", [LinearElasticity])
def test_dimensionality_check(material_model: MaterialProblem) -> None:

    default_setup, default_parameters = material_model.default_parameters()

    with pytest.raises(ValueError):
        default_parameters["g"] = 3 * ureg("m")  # gravity should be m/s²
        fem_problem = material_model(default_setup, default_parameters)


@pytest.mark.parametrize("material_model", [LinearElasticity])
def test_default_parameters(material_model: MaterialProblem) -> None:
    """This function tests if the default_parameters are complete"""

    default_setup, default_parameters = material_model.default_parameters()

    try:
        fem_problem = material_model(default_setup, default_parameters)
        fem_problem.solve()
    except KeyError:
        print("default parameter dictionary is wrong")
        raise ValueError


# problems which need material law as additional argument (from fenics-constitutive)
@pytest.mark.parametrize("material_model", [FenicsConstitutive, ]) #ConcreteAMFC
def test_dimensionality_check_fc(material_model: MaterialProblem) -> None:

    default_setup, default_parameters = material_model.default_parameters()   
    default_material_law = material_model.default_material()

    with pytest.raises(ValueError):
        default_parameters["g"] = 3 * ureg("m")  # gravity should be m/s²
        fem_problem = material_model(default_setup, default_parameters, default_material_law)


@pytest.mark.parametrize("material_model", [FenicsConstitutive, ]) #ConcreteAMFC
def test_default_parameters_fc(material_model: MaterialProblem) -> None:
    """This function tests if the default_parameters are complete"""


    default_setup, default_parameters = material_model.default_parameters() 
    default_material_law = material_model.default_material()  
    
    try:
        fem_problem = material_model(default_setup, default_parameters, default_material_law)
        fem_problem.solve()
    except KeyError:
        print("default parameter dictionary is wrong")
        raise ValueError
