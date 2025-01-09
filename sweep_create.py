# Import the W&B Python Library and log into W&B
import wandb

wandb.login()


# 2: Define the search space
import yaml
sweep_configuration = yaml.safe_load(open("sweep.yaml"))['sweep']


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, 
                       project="SSF")
print("Sweep ID: ", sweep_id)
with open("sweep_id.txt", "w") as f:
    f.write(sweep_id)