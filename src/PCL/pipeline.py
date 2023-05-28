from typing import Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .kmeans import kmeans
from random import randint
from .momentum import Momentum


class PCL(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_iters: int = 10,
        num_cluster_iters: int = 3,
        temperature: float = 0.1,
        concentration: float = 0.1,
        momentum: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.moment_state: Dict | None = None
        self.train_state: Dict | None = None
        self.num_iters = num_iters
        self.num_cluster_iters = num_cluster_iters
        self.t = temperature
        self.phi = concentration
        self.momentum = momentum
        self.moment: Momentum | None = None

    def backward(self, x: Tensor) -> Tensor:
        loss = torch.zeros(1, device=x.device).type_as(x)
        if self.moment is None or self.moment_state is None or self.train_state is None:
            self.moment_state = self.encoder.state_dict()
            self.train_state = self.encoder.state_dict()
            self.moment = Momentum(self.encoder, self.encoder, momentum=self.momentum)
        else:
            self.moment.step()
        self.encoder.train()
        for _ in range(self.num_cluster_iters):
            self.encoder.load_state_dict(self.moment_state)
            feats: torch.Tensor = self.encoder(x)
            feats = feats.flatten(end_dim=-2)
            D = feats.shape[-1]
            num_clusters = randint(2, feats.shape[-2] // 2)
            clusters, centroids, n_points = kmeans(feats, num_clusters, self.num_iters)
            dists = (feats.view(-1, D) - centroids.view(-1, D)[clusters.view(-1)]) ** 2
            dists = torch.scatter_add(
                torch.zeros(num_clusters, D).to(feats.device),
                0,
                clusters.view(-1, 1).repeat(1, D),
                dists,
            )
            p = torch.exp(-dists / self.phi)
            Q = p / torch.sum(p)
            entropy = Q * torch.log(Q + 1e-6) / self.num_cluster_iters
            entropy = entropy.mean()
            entropy.backward()
            loss += entropy

        self.encoder.load_state_dict(self.train_state)

        feats = self.encoder(x)

        clusters, centroids, n_points = kmeans(feats, num_clusters, self.num_iters)
        dists = (feats.view(-1, D) - centroids.view(-1, D)[clusters.view(-1)]) ** 2
        dists = torch.scatter_add(
            torch.zeros(num_clusters, D).to(feats.device),
            0,
            clusters.view(-1, 1).repeat(1, D),
            dists,
        )

        p = torch.exp(-dists / self.t)
        nll = -torch.log(p + 1e-6).mean()
        nll.backward()
        loss += nll
        return loss
