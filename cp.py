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

def calc_length_coverage(probs, range_vals, percentile_val, true_label):
    intervals = find_intervals(probs, percentile_val)
    if len(intervals) == 0:
        return 1, torch.tensor(range_vals[-1] - range_vals[0])
    else:
        length = 0
        cov_val = 0
        for interval in intervals:
            length += range_vals[interval[1]]- range_vals[interval[0]]
            if range_vals[interval[1]]  >= true_label and true_label >= range_vals[interval[0]]:
                cov_val =1
        return cov_val, length
    
def get_cp(args, range_vals, X_val, y_val, model):
    step_val = (max(range_vals) - min(range_vals))/len(range_vals)
    indices = (((y_val - min(range_vals)))/step_val).to(torch.int)
    indices[indices == len(range_vals)] = indices[indices == len(range_vals)] - 1
    scores = torch.nn.functional.softmax(model(X_val), dim=1)
    all_scores = scores[torch.arange(len(X_val)), indices.long()]

    
    alpha = args.alpha
    lengths = []
    coverages = []
    for i in range(len(X_val)):
        percentile_val = percentile_excluding_index(all_scores, i, alpha)
        coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])
        coverages.append(coverage)
        lengths.append(length)
    return np.mean(coverages).item(), np.std(coverages).item(), torch.mean(torch.stack(lengths)).item(), torch.std(torch.stack(lengths)).item()