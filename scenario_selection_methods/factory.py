"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Provides a single factory function that maps string method keys to
    instantiated selector objects. All nine SBSSMs are registered here;
    adding a new method requires only a new import and an additional branch.

    - key role: Decouples the evaluation loop in main.py from concrete
                selector classes; method selection is controlled entirely
                by the string keys in ``method_to_evaluations``.
    - dependency: selector_random, selector_idw, selector_gpr, selector_ipso,
                  selector_ann, selector_nndv, selector_gdnnas
    - output: Configured selector instance (BaseSelector or BaseModelSelector).
"""

from selector_random import RandomSelector, SobolSelector, LatinHypercubeSelector
from selector_gpr import GPSTLSelector
from selector_ann import ANNSelector
from selector_is import ImportanceSamplingSelector
from selector_atslg import ATSLGSelector, StandardGPRSelector

def get_selector(name: str, space, n_select: int):
    """
    Instantiate and return the selector identified by ``name``.

    Parameters
    ----------
    name : str
        Method key (case-insensitive for most entries). Supported values:
        ``"random"``, ``"sobol"``, ``"lhs"``, ``"idw"``, ``"gpr"``,
        ``"ipso"``, ``"ann"``, ``"nndv"``, ``"gdnnas"``.
    space : Space
        The criticality space passed to the selector constructor.
    n_select : int
        Sample budget passed to the selector constructor.

    Returns
    -------
    BaseSelector or BaseModelSelector
        Configured selector ready to call ``select()`` or ``get_model()``.

    Raises
    ------
    ValueError
        If ``name`` does not match any registered method.
    """
    if name.lower() == "random":
        return RandomSelector(space, n_select)
    elif name == "sobol":
        return SobolSelector(space, n_select)
    elif name == "lhs":
        return LatinHypercubeSelector(space, n_select)
    elif name == "gpr":
        return GPSTLSelector(space, n_select)
    elif name.lower() == "ann":
        return ANNSelector(space, n_select)
    elif name == "atslg":
        return ATSLGSelector(space, n_select)
    elif name == "atslgnc":
        return StandardGPRSelector(space, n_select)
    elif name == "is":
        return ImportanceSamplingSelector(space, n_select)
    else:
        raise ValueError(f"Unknown selection method: {name}")
    