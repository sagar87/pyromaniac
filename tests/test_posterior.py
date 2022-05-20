import numpy as np

from pyromaniac.posterior import make_array


def test_make_array():
    """
    GIVEN
    WHEN get_quote is called
    THEN random quote from quotes is returned
    """
    res = make_array(5)
    assert type(res) == np.ndarray
    pass
