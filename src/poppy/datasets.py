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
                torch.tensor(list(map(float, sample[:768]))),
                torch.tensor(list(map(float, sample[768:]))),
            ))
    return samples


def save_samples(location: str, samples: Iterable[Sample]):
    with open(location, mode='w', newline='', encoding='utf-8') as fp:
        samplewriter = csv.writer(fp)
        for sample in samples:
            samplewriter.writerow(sample.tolist())
    return samples


if __name__ == '__main__':

    file = open('./sample/momotaro.csv', 'r', encoding='utf-8')

    samples = []

    for sample in csv.reader(file):
        samples.append(Sample(
            bertify(sample[-1]),
            torch.tensor(list(map(float, sample[:3]))),
        ))

    file.close()

    save_samples('./sample/momotaro-params.csv', samples)
