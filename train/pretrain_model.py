import numpy as np
import torch

from modules import DynamicModel
if __name__ == "__main__":
    input_state_dim = 12
    output_state_dim = 9
    action_dim = 9
    hidden_dim=512
    model = DynamicModel(input_state_dim=input_state_dim, action_dim=action_dim, output_state_dim=output_state_dim, hidden_dim=hidden_dim)