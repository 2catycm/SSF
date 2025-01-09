# Import the W&B Python Library and log into W&B
import wandb

wandb.login()

# sweep_id = ""
with open("sweep_id.txt", "r") as f:
    sweep_id = f.read()
from train import main
wandb.agent(sweep_id, function=main,
            # count=10, 
            project="SSF"
            )