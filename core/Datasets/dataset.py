import glob
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split


def get_datasets(cfg):
    """
    cfg is the full OmegaConf object from Hydra
    """
    search_path = f"{cfg.data.root}/preprocessed/*/*.npz"
    files = glob.glob(search_path)

    train_samples, test_samples = train_test_split(
        files,
        test_size=0.05,
        random_state=42
    )

    train_dataset = instantiate(cfg.data, samples=train_samples)
    test_dataset = instantiate(cfg.data, samples=test_samples)

    return train_dataset, test_dataset