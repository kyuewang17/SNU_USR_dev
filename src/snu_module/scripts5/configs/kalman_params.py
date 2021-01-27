"""
Object Recognition Module (SNU Integrated Module)
for Outdoor Surveillance Robots

    - Kalman Filter Parameters

"""
import numpy as np


class KALMAN_PARAMS(object):
    def __init__(self, agent_type="undefined"):
        # Agent Type
        self.agent_type = agent_type

        # State Transition Matrix (Motion Model)
        self.A = np.float32([[1, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]])

        # Unit Transformation Matrix
        self.H = np.float32([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]])

        # Error Covariance Matrix
        self.P = np.float32([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]]) * 1e-6

        # State Covariance Matrix
        self.Q = np.float32([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]]) * 1e-6

        # Measurement Covariance Matrix
        self.R = np.float32([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]]) * 1e-6

        # Kalman Gain Matrix
        self.K = np.float32([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]])

    def __repr__(self):
        return self.agent_type

    def get_kalman_params(self):
        return {"A": self.A, "H": self.H, "P": self.P, "Q": self.Q, "R": self.R, "K": self.K}

    def get_kalman_params_as_list(self):
        return {
            "A": self.A.tolist(), "H": self.H.tolist(), "P": self.P.tolist(),
            "Q": self.Q.tolist(), "R": self.R.tolist(), "K": self.K.tolist()
        }

    def __call__(self, **kwargs):
        # Assertion regarding agent type
        agent_type = kwargs.get("agent_type")
        if agent_type is not None:
            assert isinstance(agent_type, str)
            if agent_type.lower() not in ["static", "dynamic"]:
                raise AssertionError()
            self.agent_type = agent_type.lower()

        # Kalman Filter Parameter Adaptation
        A, P, Q, R = kwargs.get("A"), kwargs.get("P"), kwargs.get("Q"), kwargs.get("R")
        if A is not None:
            if isinstance(A, np.ndarray):
                self.A = A
            elif isinstance(A, str):
                # TODO: Motion model adaptation via word string
                raise NotImplementedError("Not implemented yet...!")
        if P is not None:
            if isinstance(P, np.ndarray):
                self.P = P
            elif isinstance(P, float):
                self.P *= P
            else:
                raise AssertionError("Input argument 'P' must be a < ndarray > or < float > type...!")
        if Q is not None:
            if isinstance(Q, np.ndarray):
                self.Q = Q
            elif isinstance(Q, float):
                self.Q *= Q
            else:
                raise AssertionError("Input argument 'P' must be a < ndarray > or < float > type...!")
        if R is not None:
            if isinstance(R, np.ndarray):
                self.R = R
            elif isinstance(R, float):
                self.R *= R
            else:
                raise AssertionError("Input argument 'P' must be a < ndarray > or < float > type...!")

        # Return Options
        return_type = kwargs.get("return_type")
        if return_type is None or return_type == "numpy":
            return self.get_kalman_params()
        elif return_type == "list":
            return self.get_kalman_params_as_list()
        else:
            raise AssertionError()


if __name__ == "__main__":
    KPARAMS = KALMAN_PARAMS()

    pass
