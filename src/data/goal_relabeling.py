"""
Contains goal relabeling written in numpy.

Each relabeling function takes a trajectory with keys "obs_images".
It returns a new trajectory with the added keys "goal_images".

"""

import numpy as np


def uniform(traj, offset=1, future_step=1):
    """
    Relabels with a true uniform distribution over future states.
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + offset:].
    """
    next_obs_images = traj["obs_images"][1:]
    traj_len = next_obs_images.shape[0]

    rand = np.random.uniform(size=traj_len)
    low = np.arange(traj_len, dtype=np.float32) + offset
    scope = np.maximum(0, traj_len - low)
    goal_idxs = (rand * scope + low).astype(np.int64)

    goal_idxs = np.minimum(goal_idxs, traj_len - 1)

    # select goals
    traj["goal_images"] = next_obs_images[goal_idxs]

    future_idxs = np.arange(traj_len) + future_step
    future_idxs = np.minimum(future_idxs, goal_idxs)
    # select near-future
    traj["future_images"] = next_obs_images[future_idxs]

    return traj


def last_state_upweighted(traj, offset=1, future_step=1):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range
    [i + offset, i + traj_len). It then gets clipped to be less than traj_len. 
    Therefore, the first transition (i = 0) gets a goal sampled uniformly from the future,
    but for i > 0 the last state gets more and more upweighted.
    """
    next_obs_images = traj["obs_images"][1:]
    traj_len = next_obs_images.shape[0]

    # select a random future index for each transition
    offsets = np.random.uniform(
        size=traj_len,
        minval=offset,
        maxval=traj_len,
    ).astype(np.int64)

    # convert from relative to absolute indices
    goal_idxs = np.arange(traj_len) + offsets

    goal_idxs = np.minimum(goal_idxs, traj_len - 1)

    # select goals
    traj["goal_images"] = next_obs_images[goal_idxs]

    future_idxs = np.arange(traj_len) + future_step
    future_idxs = np.minimum(future_idxs, goal_idxs)
    # select near-future
    traj["future_images"] = next_obs_images[future_idxs]

    return traj

def last_k_uniform(traj, offset=10, future_step=4):
    """
    Relabels with a true uniform distribution over the last k states.
    observations[i] gets a goal sampled uniformly from the set
    next_observations[-offset:].
    """
    next_obs_images = traj["obs_images"][1:]
    traj_len = next_obs_images.shape[0]

    goal_idxs = np.random.uniform(traj_len-offset, traj_len, size=traj_len).astype(np.int64)

    goal_idxs = np.where(
        goal_idxs < np.arange(traj_len),
        traj_len-1,
        goal_idxs
    )

    # select goals
    traj["goal_images"] = next_obs_images[goal_idxs]

    future_idxs = np.arange(traj_len) + future_step
    future_idxs = np.minimum(future_idxs, goal_idxs)
    # select near-future
    traj["future_images"] = next_obs_images[future_idxs]

    return traj


GOAL_RELABELING_FUNCTIONS = {
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
    "last_k_uniform": last_k_uniform,
}
