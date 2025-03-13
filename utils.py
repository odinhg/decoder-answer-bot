import torch

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_config(config):
    print("Using configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("*" * 100)

def sample_greedy():
    pass

def sample_top_p():
    pass
