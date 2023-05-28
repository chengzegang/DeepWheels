from typing import Tuple
from . import train
import click


@click.command()
@click.option("--model-name", "-m", type=str, default="resnet18")
@click.option("--pretrained", is_flag=True)
@click.option("--datadir", "-d", type=str, required=True)
@click.option("--batch-size", type=int, default=1)
@click.option("--num-workers", type=int, default=0)
@click.option("--device", type=str, default="cuda")
@click.option("--lr", type=float, default=2e-4)
@click.option("--weight-decay", type=float, default=1e-5)
@click.option("--betas", type=tuple[float, float], default=(0.9, 0.95))
@click.option("--total-epochs", type=int, default=10)
@click.option("--num-cluster-iters", type=int, default=5)
@click.option("--num-kmeans-iters", type=int, default=10)
@click.option("--feature-size", type=int, default=768)
@click.option("--temporature", "-t", type=float, default=0.1)
@click.option("--concentration", "-p", type=float, default=0.1)
@click.option("--momentum", type=float, default=0.9)
@click.option("--image-shape", type=tuple[int, int], default=(64, 128))
@click.option("--verbose", type=bool, default=True)
def run(**kwargs):
    train._start_cli(**kwargs)


if __name__ == "__main__":
    run()
