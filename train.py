import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(model)

    g_loss = instantiate(config.g_loss).to(device)
    d_loss = instantiate(config.d_loss).to(device)
    metrics = instantiate(config.metrics)

    g_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    d_trainable_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())

    g_optimizer = instantiate(config.g_optimizer, params=g_trainable_params)
    d_optimizer = instantiate(config.d_optimizer, params=d_trainable_params)
    
    g_scheduler = instantiate(config.g_scheduler, optimizer=g_optimizer)
    d_scheduler = instantiate(config.d_scheduler, optimizer=d_optimizer)

    trainer = Trainer(
        model=model,
        losses=(g_loss, d_loss),
        metrics=metrics,
        optimizers=(g_optimizer, d_optimizer),
        schedulers=(g_scheduler, d_scheduler),
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=config.trainer.epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
