import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import bisect

def find_closest_element(sorted_list, target):
    index = bisect.bisect_left(sorted_list, target)
    
    if index == 0:
        return 0
    if index == len(sorted_list):
        return len(sorted_list) -1
    
    before = sorted_list[index - 1]
    after = sorted_list[index]
    
    if after - target < target - before:
        return index
    else:
        return index - 1
    
def max_entropy_distribution(moments, num_points=10000):
    num_moments = len(moments)
    x = np.linspace(-5, 5, num_points)
    
    # Define the probability distribution as a variable
    p = cp.Variable(num_points)
    
    # Define the objective function (negative entropy)
    objective = cp.sum(cp.entr(p))
    
    # Define the moment constraints with tolerance
    constraints = []
    for n, moment in enumerate(moments):
        constraints.append(cp.sum(cp.multiply(p ,cp.power(x, n+1))) <= moment + 1e-5)
        constraints.append(cp.sum(cp.multiply(p ,cp.power(x, n+1))) >= moment - 1e-5)
    constraints.append(cp.sum(p) == 1)
    nonnegativity_constraint = p >= 0
    constraints.append(nonnegativity_constraint)
    
    # Define and solve the optimization problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()

    # Extract the optimal probability distribution
    max_entropy_dist = p.value
    return x, max_entropy_dist


def get_px(x, moments, num_points=10000):
    all_vals, probabilities = max_entropy_distribution(moments, num_points=num_points)
    best_matching_index = find_closest_element(all_vals, x)
    return probabilities[best_matching_index]

def get_above_quantile(threshold, moments, num_points=10000):
    all_vals, probabilities = max_entropy_distribution(moments, num_points=num_points)
    return all_vals[np.where(np.asarray(probabilities >= threshold))]
