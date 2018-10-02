from builtins import range
import numpy as np

class EqualSumCombinations:
    """Class to generate all combinations of non-negative integers that have
    equal sum.

    """
    def __init__(self, sum_, size):
        """Constructor to initialize the variables.

        Parameters
        ----------
        sum_ : int
            Desired sum (must be greater than 0).
        size : int
            Number of integers (must be greater than 1).

        Raises
        ------
        ValueError
            If sum is less than 2.
        """
        if sum_ <= 0:
            raise ValueError("Sum must be positive.")
        if size < 2:
            raise ValueError("Size must be greater than 1.")

        self.dp = np.zeros((sum_+1, size+1), dtype=object)
        self.combs = self.get_combinations(sum_, size)

    def get_combinations(self, sum, n):
        """A recursive function to generate the list of all non-negative
        integer combinations of a given size that have a given sum.

        Parameters
        ----------
        sum : int
            Desired sum.
        n : int
            Desired size.

        Returns
        -------
        type : array-like
            A list containing the lists of combinations.

        """

        # Check if the list of lists has already been generated. If yes,
        # return the value.
        if self.dp[sum][n]:
            return self.dp[sum][n]

        # Initialize list and check special cases first.
        tmp_list = []
        if n == 1:
            tmp_list = [[sum]]
        elif sum == 0:
            tmp_list = [[0] * n]
        else:
            for i in range(sum, -1, -1):
                for l in self.get_combinations(sum - i, n - 1):
                    tmp_list.append([i]+l)

        # Store value into the numpy array before returning it.
        self.dp[sum][n] = tmp_list
        return tmp_list