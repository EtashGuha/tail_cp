import torch
import numpy as np

def percentile_excluding_index(vector, i, percentile):
        # Remove the value at the i-th index
        modified_vector = torch.cat((vector[:i], vector[i+1:]))

        # Calculate the percentile on the modified vector
        percentile_value = torch.quantile(modified_vector, percentile)
        
        return percentile_value
def find_intervals(data, cutoff):
    intervals = []
    start = None

    for i, item in enumerate(data):
        if item >= cutoff:
            if start is None:
                start = i
        elif start is not None:
            if i - 1 == start:
                intervals.append((start, i))
            else: 
                intervals.append((start, i-1))
            start = None

    # If the last interval extends to the end of the list
    if start is not None:
        if start == len(data) - 1:
            intervals.append((start - 1, start))
        else:
            intervals.append((start, len(data) - 1))

        

    return intervals
def get_cp(args, range_vals, X_val, y_val, model):

    step_val = (max(range_vals) - min(range_vals))/len(range_vals)
    indices = (((y_val - min(range_vals)))/step_val).to(torch.int)
    indices[indices == len(range_vals)] = indices[indices == len(range_vals)] - 1
    scores = torch.nn.functional.softmax(model(X_val), dim=1)
    all_scores = scores[torch.arange(len(X_val)), indices.long()]

    
    alpha = args.alpha
    lengths = []
    coverage = []
    for i in range(len(X_val)):
        percentile_val = percentile_excluding_index(all_scores, i, alpha)
        intervals = find_intervals(scores[i], percentile_val)
        length = 0
        cov_val = 0
        for interval in intervals:
            length += range_vals[interval[1]]- range_vals[interval[0]]
            if range_vals[interval[1]]  >= y_val[i] and y_val[i] >= range_vals[interval[0]]:
                cov_val =1
        coverage.append(cov_val)
        lengths.append(length)
    return np.mean(coverage).item(), np.std(coverage).item(), torch.mean(torch.stack(lengths)).item(), torch.std(torch.stack(lengths)).item()