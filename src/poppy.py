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

from flask import Flask
from flask import jsonify, request
from flask_cors import CORS
from typing import Iterable
from transformers import BertJapaneseTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
import torch
import os
import poppy_sample as sample


BERT_TOKENIZER = \
    BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

BERT = \
    BertModel.from_pretrained('cl-tohoku/bert-base-japanese')


def bertify(text: str):
    """
    Convert text into features using the BERT model.
    """
    inputs = BERT_TOKENIZER(
        text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = BERT(**inputs)
    return outputs.last_hidden_state[:, 0, :]


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

        self._model = nn.Sequential(
            nn.Linear(n := BERT.config.hidden_size, n*2),
            nn.Tanh(),
            nn.Linear(n*2, 1),
            nn.Tanh(),
        )

    def predict(self, text: str):
        """
        Infers the output using BERT and the Poppy model.
        """
        return self._model(bertify(text))[0, 0].item()

    def fit(self, sample: sample.Sample):
        """
        Trains the model (excluding the BERT part). It learns a mapping
        between text and an expected score.
        """
        self._model.train()

        optimizer = optim.Adam(self._model.parameters(), lr=0.0001)
        optimizer.zero_grad()

        lossfunc = nn.MSELoss()

        score = torch.tensor([sample.score], dtype=torch.float32).view(1, 1)

        pred_score = self._model(bertify(sample.text)).view(-1, 1)

        loss = lossfunc(pred_score, score)
        loss.backward()

        optimizer.step()

        return loss.item()

    def fit2(self, samples: Iterable[sample.Sample]):
        """
        Trains the model (excluding the BERT part) on a batch of samples.
        Each sample is a tuple of (text, score).
        """

        self._model.train()

        optimizer = optim.Adam(self._model.parameters(), lr=0.0001)
        optimizer.zero_grad()

        lossfunc = nn.HuberLoss()
        loss_sum = 0.0

        for text, score in samples:
            score = torch.tensor(
                [float(score)], dtype=torch.float32).view(1, 1)
            pred_score = self._model(bertify(text)).view(-1, 1)
            loss = lossfunc(pred_score, score)
            loss.backward()
            loss_sum += loss.item()

        optimizer.step()

        return loss_sum

    def save(self, location: str):
        torch.save(self._model.state_dict(), location)

    def load(self, location: str):
        if os.path.exists(location):
            torch.serialization.add_safe_globals([nn.Sequential])
            self._model.load_state_dict(
                torch.load(location, weights_only=True))
        else:
            raise FileNotFoundError(f"No such file or directory: '{location}'")


def model_init(model: Model, args: dict):

    model_location = args.setdefault(
        'model-location', './model/poppy-beta.pth')

    sample_location = args.setdefault(
        'sample-location', './sample/sample-20240730.csv')

    try:
        model.load(model_location)
    except FileNotFoundError:
        print("Model file not found. Training a new model...")

        samples = sample.loads(sample_location)

        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = model.fit2(samples)
            print(f'\rLoading now ... {epoch+1}/{num_epochs}'
                  f' - Loss: {epoch_loss:.4f}', end='')

        print()

        model.save(model_location)


model = Model()

app = Flask(__name__)
CORS(app)


@app.route('/api/beta/predict', methods=['POST'])
def _api_beta_predict():

    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No sample provided!'})

    return jsonify({'score': model.predict(text)})


if __name__ == '__main__':

    args = {}
    model_init(model, args)

    app.run(host='localhost', port=8080)
