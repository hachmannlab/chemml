"""
This module contains parent classes for the trained models.

"""

import numpy as np

def check_array_input(X, name, ndim_must_be, shape_must_be):
    """
    This method checks the dimension and shape of an input array.

    Parameters
    ----------
    X : ndarray
        The input numpy array

    name : str
        The variable name in the main method.

    ndim_must_be : int
        The required dimension of the input array.

    shape_must_be : tuple
        The required shape of the input array.

    Returns
    -------
    bool
        If True, this function confirms the type, dimension and shape of the input array are as required.

    str
        Any message raised after doing this analysis.

    """
    # type
    if isinstance(X, np.ndarray):
        # dimension
        if X.ndim == ndim_must_be :
            # length of shapes must be same
            if len(X.shape) != len(shape_must_be):
                msg = "The %iD input array %s doesn't follow the required shape %s." % (
                    ndim_must_be, name, str(shape_must_be))
                return False, msg
            # check if there are no None elements in the shape_must_be
            # if there is any replace it with actual dimension
            shape_must_be = adapt_shape_array(X, shape_must_be)
            # shape
            if X.shape != shape_must_be:
                msg = "The %iD input array %s, must have a shape of %s" % (ndim_must_be, name, str(shape_must_be))
                return False, msg
            else:
                return True, ""
        else:
            msg = "The input array %s, must be a %iD numpy array." % (name, ndim_must_be)
            return False, msg
    else:
        msg = "The input %s, must be a numpy array."%name
        return False, msg

def adapt_shape_array(X, shape_must_be):
    """
    This function takes care of the unspecified dimensions in the required shapes of data structures.
    The None elements in the shape tuple will be replaced with actual shape of the input array.

    Parameters
    ----------
    X: ndarray
        The input numpy array.

    shape_must_be: tuple
        The required shape of the input array, might contains None elements.

    Returns
    -------
    tuple
        The actual shape with no None elements.

    """
    if None in shape_must_be:
        copy_shape_must_be = list(shape_must_be)
        while None in copy_shape_must_be:
            ind = copy_shape_must_be.index(None)
            copy_shape_must_be[ind] = X.shape[ind]
        return tuple(copy_shape_must_be)
    else:
        return shape_must_be