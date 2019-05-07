import numpy as np

class FourierBasis:
    # approximate upper bound controls for how large you expect the variables to be, even though they may be inifinite
    def __init__(self, num_variables, degree, approximate_upper_bound=100):
        c_vec = np.zeros ( ((degree + 1) ** num_variables, num_variables) )

        for i in range(1, c_vec.shape[0]):
            c_vec[i] = c_vec[i - 1]
            c_vec[i, 0] += 1

            col = 0
            while c_vec[i, col] == degree + 1:
                c_vec[i, col] = 0
                col += 1
                c_vec[i, col] += 1

        self.c_vec = c_vec

        self.approximate_upper_bound = approximate_upper_bound

        # # the fancy way, with 4 state variables
        # n = degree
        # c_vec = np.zeros( (n+1, n+1, n+1, n+1, 4) )
        # for i in range(n+1):
        #     c_vec[i, :, :, :, 0] = i
        #     c_vec[:, i, :, :, 1] = i
        #     c_vec[:, :, i, :, 2] = i
        #     c_vec[:, :, :, i, 3] = i
        # c_vec = c_vec.reshape((n+1)**4, 4)


    def phi(self, state):
        norm_state = np.tanh(np.array(state) / self.approximate_upper_bound) * 2 - 1 # range of [-1, 1]

        result = np.cos(np.pi * (self.c_vec @ norm_state))
        return result
