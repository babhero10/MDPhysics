import os
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    run_dir = os.getcwd()
    writer = SummaryWriter(log_dir=run_dir)



if __name__ == "__main__":
    main()
