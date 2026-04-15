import glob
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split


def get_datasets(cfg):
    """
    cfg is the full OmegaConf object from Hydra
    """
    files = glob.glob(f"{cfg.data.root}/preprocessed/*/*.npz")

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    train_dataset = instantiate(cfg.data.dataset, samples=train_samples)
    test_dataset = instantiate(cfg.data.dataset, samples=test_samples)

    return train_dataset, test_dataset