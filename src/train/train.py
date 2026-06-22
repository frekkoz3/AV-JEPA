r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import yaml

from src.jepa.transformers import VisualTransformer, Transformer
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.game.snake import SnakeEnv, TOTAL_HEIGHT, WIDTH
from src.jepa.e2e_jepa import *
from src.utils.utils import *
import cv2
import argparse

TOTAL_EPOCHS = 16
STEPS_PER_EPOCH = 256
BATCH_SIZE = 32
ACTION_DIM = 4
REFRESH_BUFFER = 8
IMG_SIZE = 256
EMBED_DIM = 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active E2E-JEPA Training for Snake Game")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    
    trainer = ActiveE2EJEPATrainer(
        env=SnakeEnv(**config),
        encoder=VisualTransformer(img_size=IMG_SIZE, embed_dim=EMBED_DIM, mlp_dim=16), # how does this work with non-squared images?
        predictor=Transformer(input_dim=EMBED_DIM, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM, depth=6, num_heads=8, mlp_dim=16, use_adaLN=True),
        policy=PolicyPPO(**config),
        action_dim=ACTION_DIM
    )
    
    env = SnakeEnv(render_mode="rgb_array", observation_type="image")
    x_t, _ = env.reset()
    x_t = cv2.resize(x_t, (IMG_SIZE, IMG_SIZE))
    
    for epoch in range(TOTAL_EPOCHS):
        if epoch % REFRESH_BUFFER == 0:
            trainer.buffer.refresh()
            
        for step in range(STEPS_PER_EPOCH):
            
            # Choose action actively using current model state
            a_t, _ = trainer.get_action(x_t)
            
            # Step the real environment
            x_tp1, r_t, done, _, _ = env.step(a_t)
            x_tp1 = cv2.resize(x_tp1, (IMG_SIZE, IMG_SIZE))
            
            a_t_onehot = F.one_hot(torch.tensor(a_t), num_classes=ACTION_DIM).float()
            
            if done:
                x_tp1 = env.death_state()
                x_tp1 = cv2.resize(x_tp1, (IMG_SIZE, IMG_SIZE))
                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t, a_t_onehot, r_t, x_tp1, float(done))
                x_t, _ = env.reset()
                x_t = cv2.resize(x_t, (IMG_SIZE, IMG_SIZE))
            else:
                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t, a_t_onehot, r_t, x_tp1, float(done))
                # Move forward
                x_t = x_tp1
        
        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(BATCH_SIZE, epoch, TOTAL_EPOCHS)
        if metrics:
            print(f"Epoch {epoch} Metrics -> Loss: {metrics['total_loss']:.4f} | Pred: {metrics['pred_loss']:.4f}")
