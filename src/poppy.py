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


from transformers import BertJapaneseTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
import torch


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

        self._bert_tokenizer: BertJapaneseTokenizer =\
            BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese')

        self._bert: BertModel =\
            BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

        self._this_model = nn.Sequential(
            nn.Linear(self._bert.config.hidden_size, 1),
            nn.Tanh(),
        )

        self._this_optimizer = optim.Adam(
            self._this_model.parameters(), lr=0.001)
        self._this_loss = nn.MSELoss()

    def predict(self, text: str):
        """
        Infers the output using BERT and the Poppy model.
        """

        inputs = self._bert_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self._bert(**inputs)

        return self._this_model(outputs.last_hidden_state[:, 0, :])

    def fit(self, text: str, score: float):
        """
        Trains the model (excluding the BERT part). It learns a mapping
        between text and an expected score.
        """

        inputs = self._bert_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True)

        score = torch.tensor([score], dtype=torch.float32).view(1, 1)

        self._this_model.train()
        self._this_optimizer.zero_grad()

        pred_score = self._this_model(self._bert(
            **inputs).last_hidden_state[:, 0, :])

        # Ensure pred_score and score have the same size
        pred_score = pred_score.view(-1, 1)

        loss = self._this_loss(pred_score, score)
        loss.backward()
        self._this_optimizer.step()

        return loss.item()


def main():

    model = Model()

    # Sample texts and labels
    # TODO: Externalizing data
    # TODO: Preparing more samples
    texts_and_labels = [
        ("なんでやねん！", -1.0),
        ("お前、何言うてんねん！", -1.0),
        ("まあ、適当にやってくれたらええわ。", 0.0),
        ("ご確認いただけますでしょうか？", 1.0),
        ("ありがとうございます", 1.0),
    ]

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0
        for text, pred_score in texts_and_labels:
            loss = model.fit(text, pred_score)
            epoch_loss += loss
        print(f'\rLoading now ... {epoch+1}/{num_epochs}', end='')

    for text, score in texts_and_labels:
        pred_score = model.predict(text)
        print(f"Text: {text}\n"
              f"  Predicted: {pred_score.item()}\n"
              f"  Expected: {score}")

    print("POPPYは日本語のテキストの言葉遣いを評価します！")

    text = input('\n日本語のテキストを入力してください！ (終了するには"!EXIT"を入力する)\n> ')

    while text != '!EXIT':
        print(f'\n言葉遣いスコア: {model.predict(text)[0, 0]}\n')

        text = input('\n日本語のテキストを入力してください！ (終了するには"!EXIT"を入力する)\n> ')


if __name__ == '__main__':
    main()
