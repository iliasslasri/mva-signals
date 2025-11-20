def count_n_param(model):
    return sum([p.numel() for p in model.parameters()])
