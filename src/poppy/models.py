# -*- coding: utf-8 -*-
"""
Copyright (c) 2024 TK0507 (takuyakimura0507@gmail.com)

DISCLAIMER:
This software is provided "as is", without any express or implied
warranty. In no event shall the author or copyright holder be liable for
any claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

from typing import Iterable
from poppy.datasets import Sample
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import tqdm


class Model(nn.Module):

    def __init__(self) -> None:
        super(Model, self).__init__()
        self._layer_1 = nn.Linear(768, 768 * 2)
        self._layer_2 = nn.Linear(768 * 2, 768)
        self._layer_3 = nn.Linear(768, 768)
        self._layer_4 = nn.Linear(768, 768)
        self._layer_5 = nn.Linear(768, 3)

    def forward(self, x: torch.Tensor):
        x = F.mish(self._layer_1(x))
        x = F.mish(self._layer_2(x))
        x = F.mish(self._layer_3(x))
        x = F.mish(self._layer_4(x))
        x = F.logsigmoid(self._layer_5(x))
        return x


class ModelTrainer:
    """
    The Poppy model extends the BERT model. It uses BERT features to
    add new attributes to the text.
    """

    def __init__(self, model: Model) -> None:
        """
        Initializes the model, including the BERT model and the Poppy model.
        """
        self.model = model
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.model_criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=16, gamma=0.837)

    def train_once(self, samples: Iterable[Sample]):
        """
        Trains the model (excluding the BERT part) on a batch of samples.
        Each sample is a tuple of (text, score).
        """
        self.model.train()
        self.model_optimizer.zero_grad()

        losssum = 0.0
        for sample in tqdm.tqdm(samples):
            loss = self.model_criterion(self.model(sample.ivec), sample.ovec)
            loss.backward()
            losssum += loss.item()
        self.model_optimizer.step()
        self.scheduler.step()  # Update the scheduler after each batch
        return losssum / len(samples)

    def train(self, samples: Iterable[Sample], maxcount=100):
        for _ in tqdm.tqdm(range(maxcount)):
            print('\nloss:', lossresult := self.train_once(samples))
        return lossresult


def save_model(location: str, model: Model):
    torch.save(model.state_dict(), location)


def load_model(location: str, model: Model):
    model.load_state_dict(torch.load(location, weights_only=True))
