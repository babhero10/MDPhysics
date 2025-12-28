import hydra
from omegaconf import DictConfig
from dataset.Dataset import BlurDataset
from utils.seed import set_seed
# from utils.logger import Logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    train_dataset = BlurDataset(cfg.dataset, "train")
    val_dataset = BlurDataset(cfg.dataset, "val")
    test_dataset = BlurDataset(cfg.dataset, "test")

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(train_dataset[0][0].shape)
    print(val_dataset[0][0].shape)
    print(test_dataset[0][0].shape)


if __name__ == "__main__":
    main()
