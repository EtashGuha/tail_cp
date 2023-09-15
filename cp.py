import torch
import numpy as np
import pickle 
import os

def percentile_excluding_index(vector, percentile):
        # Remove the value at the i-th index

        # Calculate the percentile on the modified vector
        percentile_value = torch.quantile(vector, percentile)
        
        return percentile_value

def find_intervals_above_value_with_interpolation(x_values, y_values, cutoff):
    intervals = []
    start_x = None
    if y_values[0] >= cutoff:
        start_x = x_values[0]
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = y_values[i], y_values[i + 1]

        if min(y1, y2) <= cutoff <= max(y1, y2):
            # Calculate the x-coordinate where the line crosses the cutoff value
            x_cross = x1 + (x2 - x1) * (cutoff - y1) / (y2 - y1)

            if x1 <= x_cross <= x2:
                if start_x is None:
                    start_x = x_cross
                else:
                    intervals.append((start_x, x_cross))
                    start_x = None

    # If the line ends above cutoff, add the last interval
    if start_x is not None:
        intervals.append((start_x, x_values[-1]))

    return intervals

def calc_length_coverage(probs, range_vals, percentile_val, true_label):
    intervals = find_intervals_above_value_with_interpolation(range_vals, probs, percentile_val)
    if len(intervals) == 0:
        return 1, torch.tensor(range_vals[-1] - range_vals[0])
    else:
        length = 0
        cov_val = 0
        for interval in intervals:
            length += interval[1] - interval[0]
            if interval[1]  >= true_label and true_label >= interval[0]:
                cov_val = 1
        return cov_val, length
    
def get_all_scores(range_vals, X, y, model):
    step_val = (max(range_vals) - min(range_vals))/(len(range_vals) - 1)
    indices_up = torch.ceil((y - min(range_vals))/step_val)
    indices_down = torch.floor((y - min(range_vals))/step_val)
    
    how_much_each_direction = (y - min(range_vals))/step_val - indices_down

    weight_up = how_much_each_direction
    weight_down = 1 - how_much_each_direction

    bad_indices = torch.where(torch.logical_or(y > max(range_vals), y < min(range_vals)))
    indices_up[bad_indices] = 0
    indices_down[bad_indices] = 0

    scores = torch.nn.functional.softmax(model(X), dim=1)
    all_scores = scores[torch.arange(len(X)), indices_up.long()] * weight_up + scores[torch.arange(len(X)), indices_down.long()] * weight_down
    all_scores[bad_indices] = 0
    return scores, all_scores
def get_cp_lists(args, range_vals, X_val, y_val,  X_cal, y_cal, model):

    _, all_scores = get_all_scores(range_vals, X_cal, y_cal, model)
    scores, _ = get_all_scores(range_vals, X_val, y_val, model)
    
    alpha = args.alpha
    lengths = []
    coverages = []
    for i in range(len(X_val)):
        percentile_val = percentile_excluding_index(all_scores, alpha)
        coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])
        coverages.append(coverage)
        lengths.append(length)

    return coverages, lengths

def get_cp(args, range_vals, X_val, y_val,  X_cal, y_cal, model):
    coverages, lengths = get_cp_lists(args, range_vals, X_val, y_val,  X_cal, y_cal, model)
    return np.mean(coverages).item(), np.std(coverages).item(), torch.mean(torch.stack(lengths)).item(), torch.std(torch.stack(lengths)).item()