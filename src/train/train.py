import yaml
import time
import cv2 #will need for registration maybe
import argparse
import numpy as np
import uuid
import torch
import shutil
from pathlib import Path

from src.jepa.transformers import VisualTransformer, Transformer
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, WIDTH, CELL_SIZE, BAR_HEIGHT
from src.jepa.e2e_jepa import *
from src.utils.utils import *

ACTION_DIM = 4
EMBED_DIM = 64

GPU = "cuda"
CPU = "cpu"
XPU = "xpu"

RUN_NAME = f"{time.time()} - {uuid.uuid1()}"
DEFAULT_SAVE_LOCATION = f"models/e2e/{RUN_NAME}/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active E2E-JEPA Training for Snake Game")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    source = config_path
    destination = f"models/e2e/{RUN_NAME}/config.yaml"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination) # we preserve the exact configuration

    # Training parameters
    DEVICE = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    TOTAL_EPOCHS = config.get("n_epochs", 200)
    STEPS_PER_EPOCH = config.get("steps_per_epoch", 256)
    BATCH_SIZE = config.get("batch_size", 32)
    REFRESH_BUFFER = config.get("refresh_buffer_freq", 8)
    WHERE_SAVE = config.get("save_path", DEFAULT_SAVE_LOCATION)
    EPOCHS_PER_CHECKPOINT = config.get("epochs_per_checkpoint", TOTAL_EPOCHS//2)
    CLEAN_CHECKPOINTS = config.get("clean_checkpoints", True)

    # Environment parameters
    ACTION_DIM = config.get("action_dim", 4)
    CELL_SIZE = config.get("cell_size", 20)
    GRID_WIDTH, GRID_HEIGHT = config.get("grid_width", 20), config.get("grid_height", 20)
    WIDTH = GRID_WIDTH * CELL_SIZE
    TOTAL_HEIGHT = GRID_HEIGHT * CELL_SIZE + BAR_HEIGHT
    IMG_SIZE = (WIDTH, TOTAL_HEIGHT, 3)
    N_OBSTACLES = config.get("n_obstacles", 10)
    FPS = config.get("fps", 10)

    # Encoder parameters
    embed_dim = config.get("embedding_dim", 64)
    enc_mlp_dim = config.get("enc_mlp_dim", 256)
    enc_n_heads = config.get("enc_n_heads", 4)
    enc_depth = config.get("enc_depth", 3)
    enc_patch_size = config.get("enc_patch_size", 6)
    enc_patch_in_channels = config.get("enc_path_in_channels", 3)

    # Predictor parameters
    pred_hidden_dim = config.get("pred_hidden_dim", 64)
    pred_cond_dim = config.get("pred_cond_dim", 1)
    pred_mlp_dim = config.get("pred_mlp_dim", 256)
    pred_n_heads = config.get("pred_n_heads", 4)
    pred_depth = config.get("pred_depth", 3)
    use_adaLN = config.get("use_adaLN", True)
    dropout = config.get("dropout", 0.0)

    trainer = ActiveE2EJEPATrainer(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=IMG_SIZE,
                                  embed_dim=embed_dim,
                                  patch_size=CELL_SIZE,
                                  mlp_dim=enc_mlp_dim,
                                  num_heads=enc_n_heads,
                                  depth=enc_depth).to(device=DEVICE),
        predictor=Transformer(input_dim=embed_dim,
                              hidden_dim=pred_hidden_dim,
                              cond_dim=pred_cond_dim,
                              output_dim=embed_dim,
                              depth=pred_depth,
                              num_heads=pred_n_heads,
                              mlp_dim=pred_mlp_dim,
                              use_adaLN=use_adaLN).to(device=DEVICE),
        policy=eval(config["pol_type"])(**config),
        action_dim=ACTION_DIM,
        embed_dim=embed_dim
    )
    
    env = SnakeEnv(render_mode="rgb_array", observation_type="image")
    x_t, _ = env.reset()
    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=DEVICE)

    for epoch in range(TOTAL_EPOCHS):
        if epoch % REFRESH_BUFFER == 0:
            trainer.buffer.refresh()

        trainer.encoder.eval()
        trainer.predictor.eval()
        
        with torch.no_grad():

            for step in range(STEPS_PER_EPOCH):
                z_t = trainer.encoder(x_t)[:, 0, :]
                
                # Choose action actively using current model state
                a_t, _ = trainer.get_action(z_t.detach().unsqueeze(0))
                
                # Step the real environment
                x_tp1, r_t, done, _, info = env.step(a_t)

                # Check if the action is actually legit
                if info:
                    a_t = info["act"]

                x_tp1 = torch.tensor(np.expand_dims(x_tp1, 0)).float().to(device=DEVICE)

                z_tp1 = trainer.encoder(x_tp1)[:, 0, :]

                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t.squeeze(0).to(device=CPU), torch.tensor(a_t).float().to(device=CPU), r_t, x_tp1.squeeze(0).to(device=CPU), float(done))
                
                if done:
                    # Reset        
                    x_t, _ = env.reset()
                    x_t = torch.tensor(np.expand_dims(x_t, 0)).float().to(device=DEVICE)
                else:
                    # Move forward
                    x_t = x_tp1
            
        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(BATCH_SIZE, epoch, TOTAL_EPOCHS, device=DEVICE)
        if metrics:
            print(f"Epoch {epoch} Metrics -> Loss: {metrics['total_loss']:.8f} | Pred: {metrics['pred_loss']:.8f} | Policy : {metrics['policy_loss']:.8f}")

        # Dynamically saving checkpoints and removing them
        if epoch%EPOCHS_PER_CHECKPOINT == 0:
            save_results(f"{WHERE_SAVE}{epoch//EPOCHS_PER_CHECKPOINT}.pkl", trainer.predictor, trainer.encoder, trainer.policy.network)
            if CLEAN_CHECKPOINTS:
                old = Path(f"{WHERE_SAVE}{epoch//EPOCHS_PER_CHECKPOINT - 1}.pkl")
                if old.exists():
                    old.unlink()
    
    if CLEAN_CHECKPOINTS:
        import math
        # Removing the last checkpoint searching for the correct index with floor and ceil functions
        old_ceil = Path(f"{WHERE_SAVE}{math.ceil(TOTAL_EPOCHS//EPOCHS_PER_CHECKPOINT)}.pkl")
        old_floor = Path(f"{WHERE_SAVE}{math.floor(TOTAL_EPOCHS//EPOCHS_PER_CHECKPOINT)}.pkl")
        if old_ceil.exists():
            old_ceil.unlink()
        if old_floor.exists():
            old_floor.unlink()

    save_results(f"{WHERE_SAVE}final.pkl",  trainer.predictor, trainer.encoder, trainer.policy.network)
