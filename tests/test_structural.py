import numpy as np
import pandas as pd
import pytest
from sovereign_bond_classifier.structural import CommonComponentModel


def test_common_component_recovery():
    z = np.array([-2., -1., 0., 1., 2.])
    rho = 0.03 + 0.01 * z + np.array([.0001, -.0001, 0, .0001, -.0001])
    model = CommonComponentModel().fit(pd.DataFrame({"rho": rho, "z": z}), target="rho", features=["z"])
    assert model.common_component_ == pytest.approx(0.03, abs=0.0002)
    assert model.coefficients_[0] == pytest.approx(0.01, abs=0.0002)


def test_constant_covariate_blocks_identification():
    frame = pd.DataFrame({"rho": [.01, .02, .03, .04], "constant": [1., 1., 1., 1.]})
    with pytest.raises(ValueError, match="not separately identified"):
        CommonComponentModel().fit(frame, target="rho", features=["constant"])
