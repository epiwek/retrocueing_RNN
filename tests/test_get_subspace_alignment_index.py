import pytest
import numpy as np
import constants.constants_expt1 as c
import src.subspace_alignment_index as ai


def test_get_AI_within_delay_max_dim_not_integer():
    with pytest.raises(AssertionError):
        ai.get_AI_all_dims(c, np.zeros((8, 200)), max_dim="not_an_integer")
    with pytest.raises(AssertionError):
        ai.get_AI_all_dims(c, np.zeros((8, 200)), max_dim=3.14)
    with pytest.raises(AssertionError):
        ai.get_AI_all_dims(c, np.zeros((8, 200)), max_dim=None)
    with pytest.raises(AssertionError):
        ai.get_AI_all_dims(c, np.zeros((8, 200)), max_dim=[1, 2, 3])
