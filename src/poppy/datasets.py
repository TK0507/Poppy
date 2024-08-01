from typing import Iterable
from typing import NamedTuple
from transformers import BertJapaneseTokenizer
from transformers import BertModel
import torch
import csv


BERT_TOKENIZER = BertJapaneseTokenizer\
    .from_pretrained('cl-tohoku/bert-base-japanese')

BERT = BertModel\
    .from_pretrained('cl-tohoku/bert-base-japanese')


def bertify(text: str) -> torch.Tensor:
    """
    Convert text into features using the BERT model.
    """
    inputs = BERT_TOKENIZER(
        text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = BERT(**inputs)

    return outputs.last_hidden_state[:, 0, :].squeeze(0)


class Sample(NamedTuple):

    ivec: torch.Tensor
    ovec: torch.Tensor

    def tolist(self):
        return self.ivec.tolist() + self.ovec.tolist()


def load_samples(location: str) -> Iterable[Sample]:
    samples: list[Sample] = []
    with open(location, mode='r', encoding='utf-8') as fp:
        for sample in csv.reader(fp):
            samples.append(Sample(
                ivec=torch.tensor(
                    list(map(float, sample[:768])), dtype=torch.float32),
                ovec=torch.tensor(
                    list(map(float, sample[768:])), dtype=torch.float32),
            ))
    return samples


def save_samples(location: str, samples: Iterable[Sample]):
    with open(location, mode='w', newline='', encoding='utf-8') as fp:
        desc = csv.writer(fp)
        for sample in samples:
            desc.writerow(sample.tolist())
    return samples
