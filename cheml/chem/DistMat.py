import numpy as np
# Todo: convert it to pandas dataframe for the consistency

class DistanceMatrix(object):
    """ (distance_matrix)
    This function finds the distance between rows of data in the input_matrix.
    The distance matrix can later be introduced to a custom kernel function.

    Parameters
    ----------
    input_matrix: numpy array
    Each element (row) of input_matrix can be a vector, matrix or list of matrices.
    These three types of representation are originally based on possible outpts
    of Coulomb_Matrix function.

    norm_type: string or integer, optional (default = 'fro')
        norm_type defines the type of norm of the difference of two rows of input
        matrix. Here is a list of available norms:

              type          norm for matrices               norm for vectors
              ----          -----------------               ----------------
              'l2'          Frobenius norm                  l2 norm for vector
              'l1'          sum(sum(abs(x-y)))              -
              'nuc'         nuclear norm                    -
              'inf'         max(sum(abs(x-y), axis=1))      max(abs(x-y))
              '-inf'        min(sum(abs(x-y), axis=1))      min(abs(x-y))
              0             -                               sum(x!=0)
              1             max(sum(abs(x-y), axis=0))      sum(abs(x)**ord)**(1./ord)
              -1            min(sum(abs(x-y), axis=0))      sum(abs(x)**ord)**(1./ord)
              2             2-norm (largest singular value) sum(abs(x)**ord)**(1./ord)
              -2            smallest singular value         sum(abs(x)**ord)**(1./ord)

                            norm for list of matrices
                            -------------------------
              'avg'         avg(sum(d(x,y[l])+d(x[l],y))))
              'max'         avg(max(d(x,y[l])+d(x[l],y))))

            * most of these norms are provided by numpy.linalg.norm
    nCores: integer, optional (default = 1)
        number of cores for multiprocessing.

    Return
    ------
    distance matrix
    """
    def __init__(self,norm_type='fro', nCores=1):
        self.norm_type = norm_type
        self.nCores = nCores
        # Todo: mpi code is required for the multiprocessing

    def transform(self,input_matrix):
        distance_matrix = []
        dim = input_matrix.shape
        if len(dim) == 2:
            # vector: Ndata=dim[0];  Nvector_elements=dim[1]
            pass
        elif len(dim) == 3:
            # matrix: Ndata=dim[0]; Nmatrix_rows=dim[1]; Nmatrix_cols=dim[2]
            for i in range(dim[0]):
                vect = []
                for k in range(0, i):
                    vect.append(distance_matrix[k][i])
                for j in range(i, dim[0]):
                    if i == j:
                        vect.append(0.0)
                    else:
                        if self.norm_type in ['fro', 'nuc', 2, 1, -1, -2]:
                            vect.append(np.linalg.norm(input_matrix[i] - input_matrix[j], ord=self.norm_type))
                        elif self.norm_type == 'inf':
                            vect.append(np.linalg.norm(input_matrix[i] - input_matrix[j], ord=np.inf))
                        elif self.norm_type == '-inf':
                            vect.append(np.linalg.norm(input_matrix[i] - input_matrix[j], ord=-np.inf))
                        elif self.norm_type == 'abs':
                            vect.append(sum(sum(abs(input_matrix[i] - input_matrix[j]))))
                        else:
                            msg = "The norm_type '%s' is not a defined type of norm" % self.norm_type
                            raise ValueError(msg)
                distance_matrix.append(vect)
            return np.array(distance_matrix)
        elif len(dim) == 4:
            # list of matrices: Ndata=dim[0]; Nmatrices_per_row=dim[1]; Nmatrix_rows=dim[2]; Nmatrix_cols=dim[3]
            pass
        else:
            msg = "the structure of input_matrix is not supported"
            raise ValueError(msg)