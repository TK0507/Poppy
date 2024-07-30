from typing import NamedTuple
from typing import Iterable
import csv


class Sample(NamedTuple):

    """
    This class represents a single entry of sample data.
    """

    text: str
    score: float


def loads(filename: str) -> Iterable[Sample]:
    """
    This function loads sample data.
    The sample data contains a text column and a score column.
    """

    with open(filename, 'r', encoding='utf-8') as fp:
        samples = [Sample(text=row['text'], score=row['score'])
                   for row in csv.DictReader(fp)]
    return samples


if __name__ == '__main__':

    """
    If this file is launched as main,
    it will be executed in the test phase.
    It is not recommended to run this file outside the test phase.
    """

    for sample in loads('./sample/sample-20240730.csv'):
        print(f'text: {sample.text}')
        print(f'score: {sample.score}')
