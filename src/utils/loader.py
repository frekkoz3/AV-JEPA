r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import yaml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.jepa.transformers import VisualTransformer, Predictor, Projector
from src.jepa.e2e_jepa import *
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, GRID_HEIGHT, WIDTH, CELL_SIZE, BAR_HEIGHT, GRID_WIDTH
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.utils.utils import flat_config

from torch.optim import lr_scheduler, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR

def load_e2e(config_path : str, weights_path : str):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = flat_config(config)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Environment parameters
    action_dim = config.get("action_dim", 4)
    cell_size = CELL_SIZE
    grid_width, grid_height = WIDTH, GRID_HEIGHT
    width = WIDTH
    total_height = TOTAL_HEIGHT
    img_size = (width, total_height, 3)
    n_obstacles = config.get("n_obstacles", 10)
    fps = config.get("fps", 10)

    # Optimizer
    optimizer = config.get("optimizer", "Adam")
    lr_init = config.get("lr_init", 1e-4)
    lr_scheduler = config.get("lr_scheduler", "ExponentialLR")
    lr_step_size = config.get("lr_step_size", 10)
    lr_gamma = config.get("lr_gamma", 0.9)

    # Encoder parameters
    embed_dim = config.get("embedding_dim", 64)
    enc_mlp_dim = config.get("enc_mlp_dim", 256)
    enc_n_heads = config.get("enc_n_heads", 4)
    enc_depth = config.get("enc_depth", 3)
    enc_patch_size = config.get("enc_patch_size", 6)
    enc_patch_in_channels = config.get("enc_path_in_channels", 3)

    #Projector
    proj_hidden_dim = config.get("proj_hidden_dim")

    # Predictor parameters
    pred_hidden_dim = config.get("pred_hidden_dim", 64)
    pred_cond_dim = config.get("pred_cond_dim", 1)
    pred_mlp_dim = config.get("pred_mlp_dim", 256)
    pred_n_heads = config.get("pred_n_heads", 4)
    pred_depth = config.get("pred_depth", 3)
    use_adaLN = config.get("use_adaLN", True)
    dropout = config.get("dropout", 0.0)
    horizon = config.get("horizon", 1)

    model = E2EJEPA(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=img_size,
                                  embed_dim=embed_dim,
                                  patch_size=cell_size,
                                  mlp_dim=enc_mlp_dim,
                                  num_heads=enc_n_heads,
                                  depth=enc_depth).to(device=device),
        predictor=Predictor(embed_dim=embed_dim,
                            hidden_dim=pred_hidden_dim,
                            action_dim=action_dim,
                            depth=pred_depth,
                            num_heads=pred_n_heads,
                            mlp_dim=pred_mlp_dim,
                            use_adaLN=use_adaLN,
                            dropout=dropout).to(device=device),
        projector=Projector(embed_dim=embed_dim, hidden_dim=proj_hidden_dim)
        policy=eval(config["pol_type"])(**config),
        action_dim=action_dim,
        embed_dim=embed_dim,
        optimizer_name = optimizer,
        lr_init = lr_init,
        lr_scheduler = lr_scheduler,
        lr_gamma = lr_gamma,
        device=device,
        horizon=horizon,
        )

    load_results(weights_path, 
                 model.predictor,
                 model.encoder,
                 model.projector,
                 model.policy.network,
                 model.optimizer,
                 model.scheduler,
                 model.policy.optimizer,
                 model.policy.scheduler,
                 model.policy.epsilon_strategy)
    
    return model