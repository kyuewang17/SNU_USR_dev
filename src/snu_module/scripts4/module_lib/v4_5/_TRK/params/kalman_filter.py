"""
SNU Integrated Module v5.0
  - Code which defines Kalman Filter Parameters
  - State Vector Components are as below
    - [ x y dx dy w h D dD ] (1x8)

"""
import numpy as np
import filterpy.kalman.kalman_filter as kalmanfilter


_A = np.float32([[1, 0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1]])

_H = np.float32([[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]])

_P = np.float32([[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]])

_Q = np.float32([[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]])

_R = np.float32([[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]])


class KALMAN_FILTER(object):
    def __init__(self, **kwargs):
        # State Transition Matrix
        A = kwargs.get("A")
        self.A = _A if A is None else A

        # Unit Transformation Matrix
        H = kwargs.get("H")
        self.H = _H if H is None else H

        # Error Covariance Matrix
        P = kwargs.get("P")
        self.P = _P if P is None else P

        # State Covariance Matrix
        Q = kwargs.get("Q")
        self.Q = _Q if Q is None else Q

        # Measurement Covariance Matrix
        R = kwargs.get("R")
        self.R = _R if R is None else R

        # Kalman Gain Matrix
        self.K = np.eye(8, dtype=np.float32)

        # Prediction Matrix
        Pp = kwargs.get("Pp", np.eye(8, dtype=np.float32))
        self.Pp = Pp

        # Init Params (private variables)

    def predict(self, state):
        assert isinstance(state, np.ndarray) and state.size == 8
        state = state.reshape(-1)
        pred_state, self.Pp = kalmanfilter.predict(state, self.P, self.A, self.Q)
        return pred_state

    def update(self, pred_state, observation):
        assert isinstance(pred_state, np.ndarray) and pred_state.size == 8
        assert isinstance(observation, np.ndarray) and observation.size == 6

        state, self.P = kalmanfilter.update(pred_state, self.Pp, observation, self.R, self.H)
        return state


if __name__ == "__main__":
    pass
