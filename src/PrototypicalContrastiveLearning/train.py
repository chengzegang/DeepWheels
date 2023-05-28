from functools import partial
from typing import Tuple

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, Resize
from tqdm.auto import tqdm

from . import lr_lambda, models
from .dataset import ImageFolder
from .pipeline import PCL


def start(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cpu",
    optimizer: Optimizer | str = "adam",
    scheduler: LambdaLR | str = "cosine",
    batch_size: int = 32,
    total_epochs: int = 1,
    num_workers: int = 0,
    num_cluster_iters: int = 10,
    num_kmeans_iters: int = 10,
    temporature: float = 0.1,
    concentration: float = 0.1,
    momentum: float = 0.9,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
    if isinstance(optimizer, str):
        match optimizer:
            case "adam":
                optimizer = Adam(model.parameters(), **kwargs)
    total_steps = len(dataloader) * total_epochs
    assert isinstance(optimizer, Optimizer)
    scheduler = LambdaLR(optimizer, partial(lr_lambda.cosine_lr_lambda, 0, total_steps))
    pipeline = PCL(
        model, num_kmeans_iters, num_cluster_iters, temporature, concentration, momentum
    )
    pipeline = pipeline.to(device)
    pipeline.train()

    for epoch in (
        pbar := tqdm(range(total_epochs), disable=not verbose, desc="Training")
    ):
        for idx, image in enumerate(dataloader):
            image = image.to(device).float()
            optimizer.zero_grad()
            loss = pipeline.backward(image)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=loss.item(), epoch=epoch)

    return pipeline.encoder


def _start_cli(
    model_name: str,
    pretrained: bool,
    datadir: str,
    batch_size: int,
    num_workers: int,
    device: str,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    total_epochs: int,
    num_cluster_iters: int,
    num_kmeans_iters: int,
    feature_size: int,
    temporature: float,
    concentration: float,
    momentum: float,
    image_shape: Tuple[int, int],
    verbose: bool,
    **kwargs,
) -> nn.Module:
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = models.build(model_name, pretrained, feature_size, normalize, **kwargs)
    dataset = ImageFolder(datadir, Resize(image_shape, antialias=True))

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    total_steps = len(dataset) // batch_size * total_epochs
    scheduler = LambdaLR(optimizer, partial(lr_lambda.cosine_lr_lambda, 0, total_steps))

    return start(
        model,
        dataset,
        device,
        optimizer,
        scheduler,
        batch_size,
        total_epochs,
        num_workers,
        num_cluster_iters,
        num_kmeans_iters,
        temporature,
        concentration,
        momentum,
        verbose,
        **kwargs,
    )
