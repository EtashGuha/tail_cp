import torch
import numpy as np

def percentile_excluding_index(vector, i, percentile):
        # Remove the value at the i-th index
        modified_vector = torch.cat((vector[:i], vector[i+1:]))

        # Calculate the percentile on the modified vector
        percentile_value = torch.quantile(modified_vector, percentile)
        
        return percentile_value

def get_cp(args, range_vals, X_val, y_val, model):

    step_val = (max(range_vals) - min(range_vals))/len(range_vals)
    indices = (((y_val - min(range_vals)))/step_val).to(torch.int)

    scores = torch.nn.functional.softmax(model(X_val), dim=1)
    all_scores = scores[torch.arange(len(X_val)), indices.long()]

    
    alpha = args.alpha
    lengths = []
    coverage = []
    for i in range(len(X_val)):
        percentile_val = percentile_excluding_index(all_scores, i, alpha)
        cp_vals = range_vals[torch.where(scores[i] > percentile_val)]
        top_range, bottom_range = max(cp_vals), min(cp_vals)
        if top_range >= y_val[i] and y_val[i] >= bottom_range:
            coverage.append(1)
        else:
            coverage.append(0)
        lengths.append(top_range - bottom_range)
    return np.mean(coverage).item(), np.std(coverage).item(), torch.mean(torch.stack(lengths)).item(), torch.std(torch.stack(lengths)).item()