"""
SNU Integrated Module v5.0
  - Code which defines Association via Hungarian Algorithm
    (Linear Assignment)

"""
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian


def associate(cost_matrix, cost_thresh=None):
    """
    Cost Matrix must be a numpy array, with size of ( N x M )
    - N : # of Workers
    - M : # of Works
    - "np.nan" values in the cost matrix means that the
      indicating (worker-work) pair cannot be assigned

    """
    # Check for Cost Matrix
    assert isinstance(cost_matrix, np.ndarray) and len(cost_matrix.shape) == 2

    # Get Worker and Work Indices
    worker_indices = list(range(cost_matrix.shape[0]))
    work_indices = list(range(cost_matrix.shape[1]))

    # Detect np.nan values in cost matrix
    ignore_flag_matrix = np.isnan(cost_matrix)

    # If all components in cost matrix are np.nan, return as below
    if ignore_flag_matrix.all():
        matches = []
        unmatched_worker_indices, unmatched_work_indices = worker_indices, work_indices
        return matches, unmatched_worker_indices, unmatched_work_indices

    # If any np.nan components are detected, temporarily replace cost value with very high numeric values
    # (eventually, matching will not be available in a brute-force way for the components)
    # (this replacement is just for preventing numeric errors)
    elif ignore_flag_matrix.any():
        cost_matrix[ignore_flag_matrix] = np.nanmax(cost_matrix) * 10.0

    # Apply Hungarian Matching Algorithm (Linear Sum Assignment)
    matched_indices = np.array(hungarian(cost_matrix)).T

    # Collect Unmatched Worker Indices
    unmatched_worker_indices = []
    for worker_idx in worker_indices:
        if worker_idx not in matched_indices[:, 0]:
            unmatched_worker_indices.append(worker_idx)

    # Collect Unmatched Work Indices
    unmatched_work_indices = []
    for work_idx in work_indices:
        if work_idx not in matched_indices[:, 1]:
            unmatched_work_indices.append(work_idx)

    # Filter-out Matched with Cost lower then the threshold
    matches = []
    for m in matched_indices:
        if ignore_flag_matrix[m[0], m[1]] is True:
            unmatched_worker_indices.append(m[0])
            unmatched_work_indices.append(m[1])
        else:
            if cost_thresh is not None:
                if cost_matrix[m[0], m[1]] > cost_thresh:
                    unmatched_worker_indices.append(m[0])
                    unmatched_work_indices.append(m[1])

            # Append to Matches
            matches.append(m.reshape(1, 2))

    # Convert to Numpy Array
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # Return
    return matches, unmatched_worker_indices, unmatched_work_indices


if __name__ == "__main__":
    pass
