import pickle
with open("for_eugene.pkl", "rb") as f:
    range_vals, scores, cutoff, truth = pickle.load(f)

with open("for_eugene_numpy.pkl", "wb") as f:
    pickle.dump((range_vals.detach().numpy(), scores.detach().numpy(), cutoff.item(), truth.item()), f)