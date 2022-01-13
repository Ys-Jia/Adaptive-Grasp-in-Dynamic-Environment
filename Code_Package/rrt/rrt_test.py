from rrt_star_bid_h import RRTStarBidirectionalHeuristic
from p_utils import *
import SearchSpace

# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

X_dimensions = np.array([(-1, 1), (-1, 1), (0, 1)])  # dimensions of Search Space
# obstacles
Obstacles = np.array(
    [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
     (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
x_init = (0, 0, 0)  # starting location
x_goal = (100, 100, 100)  # goal location

Q = np.array([(0.1, 0.1, 0.1)])  # length of tree edges
r = 0.01  # length of smallest edge to check for intersection with obstacles
max_samples = 50  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRTStarBidirectionalHeuristic(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_star_bid_h()