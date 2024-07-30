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
from transformers import BertJapaneseTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
import torch
import os
import poppy_sample as sample


class Model:
    """
    The Poppy model extends the BERT model. It uses BERT features to
    add new attributes to the text.
    """

    PREDKEYS = ('wording',)

    def __init__(self) -> None:
        """
        Initializes the model, including the BERT model and the Poppy model.
        """

        self._bert_tokenizer = BertJapaneseTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese')
        self._bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

        self._this_model = nn.Sequential(
            nn.Linear(self._bert.config.hidden_size, 1),
            nn.Tanh(),
        )

        self._this_optimizer = optim.Adam(
            self._this_model.parameters(), lr=0.001)
        self._this_loss = nn.MSELoss()

    def _encode(self, text: str):
        inputs = self._bert_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self._bert(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def predict(self, text: str):
        """
        Infers the output using BERT and the Poppy model.
        """
        return self._this_model(self._encode(text))

    def fit(self, sample: sample.Sample):
        """
        Trains the model (excluding the BERT part). It learns a mapping
        between text and an expected score.
        """
        score = torch.tensor([sample.score], dtype=torch.float32).view(1, 1)

        self._this_model.train()
        self._this_optimizer.zero_grad()

        pred_score = self._this_model(self._encode(sample.text)).view(-1, 1)

        loss = self._this_loss(pred_score, score)
        loss.backward()
        self._this_optimizer.step()

        return loss.item()

    def fit2(self, samples: Iterable[sample.Sample]):
        """
        Trains the model (excluding the BERT part) on a batch of samples.
        Each sample is a tuple of (text, score).
        """
        self._this_model.train()
        self._this_optimizer.zero_grad()

        total_loss = 0.0

        for text, score in samples:
            score = torch.tensor(
                [float(score)], dtype=torch.float32).view(1, 1)
            pred_score = self._this_model(self._encode(text)).view(-1, 1)
            loss = self._this_loss(pred_score, score)
            loss.backward()
            total_loss += loss.item()

        self._this_optimizer.step()

        return total_loss

    def save(self, location: str):
        torch.save(self._this_model.state_dict(), location)

    def load(self, location: str):
        if os.path.exists(location):
            torch.serialization.add_safe_globals([nn.Sequential])
            self._this_model.load_state_dict(
                torch.load(location, weights_only=True))
        else:
            raise FileNotFoundError(f"No such file or directory: '{location}'")


def main(args: dict):

    model_location = args.setdefault(
        'model-location', './model/poppy-1.0.1.pth')

    sample_location = args.setdefault(
        'sample-location', './sample/sample-20240730.csv')

    model = Model()

    try:
        model.load(model_location)
    except FileNotFoundError:
        print("Model file not found. Training a new model...")

        samples = sample.loads(sample_location)

        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = model.fit2(samples)
            print(f'\rLoading now ... {
                  epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}', end='')

        model.save(model_location)

    print("POPPYは日本語のテキストの言葉遣いを評価します！")

    text = input('\n日本語のテキストを入力してください！ (終了するには"!EXIT"を入力する)\n> ')

    while text != '!EXIT':
        print(f'\n言葉遣いスコア: {model.predict(text)[0, 0]}\n')

        text = input('\n日本語のテキストを入力してください！ (終了するには"!EXIT"を入力する)\n> ')


if __name__ == '__main__':

    args = {}

    main(args)
