r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import torch
from src.jepa.transformers import *

model = VisualTransformer(
    img_size=(256, 412, 3),
    embed_dim=768,
    mlp_dim=3072,
    patch_size=16,
    num_heads=12,
    depth=6
)
"""
model = Transformer(
    input_dim = 10,
    hidden_dim = 5,
    output_dim = 5,
    depth = 3,
    num_heads = 1,
    mlp_dim = 6,
    dropout=0.0,
    use_adaLN=True,
)
"""
device = "cuda"

model.to(device=device)

x = torch.randn(10, 3, 412, 256).to(device = device)
# c = torch.randint(0, 4, (10, 10), dtype=torch.float32).to(device = device)

num_classes = 5

head = nn.Linear(768, num_classes)

head.to(device=device)

labels = torch.randint(0, num_classes, (10,)).to(device=device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4
)

for step in range(1000):
    optimizer.zero_grad()

    features = model(x)
    logits = head(features[:, 0])   # CLS token

    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        pred = logits.argmax(-1)
        acc = (pred == labels).float().mean()

        print(step, loss.item(), acc.item())
