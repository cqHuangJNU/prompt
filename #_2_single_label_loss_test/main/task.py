import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List
import logging
from .util import InputExample


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    对传入的数据集乱序并返回num_examples条文本数据.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples

class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        实现针对每个标签存储指定数量 max_examples 的对应文本.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples

class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

class ExampleZhProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.txt"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")

    def get_labels(self):
        return ["好", "坏"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, 'r', encoding='utf-8') as f:
            data = [line.strip().split('\t') for line in f.readlines()]
            for idx, row in enumerate(data):
                guid = "%s-%s" % (set_type, idx)
                text_a, label = row
                text_b = ''

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples

class ExampleEnProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.txt"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")

    def get_labels(self):
        return ["good", "bad"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, 'r', encoding='utf-8') as f:
            data = [line.strip().split('\t') for line in f.readlines()]
            for idx, row in enumerate(data):
                guid = "%s-%s" % (set_type, idx)
                text_a, label = row
                text_b = ''

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples

class EprstmtZhProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train_few_all.json"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev_few_all.json"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test_public.json"), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            for idx, row in enumerate(data):
                guid = "%s-%s" % (set_type, idx)
                text_a, label = row['sentence'], '0' if row['label'] == 'Positive' else '1'
                text_b = ''

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class AgnewsEnProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train_new.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test.csv"), "test")

    def get_labels(self):
        return ["1", "2", "3", "4"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = headline.replace('\\', ' ') + body.replace('\\', ' ')
                text_b = ''

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


PROCESSORS = {
    "example_zh": ExampleZhProcessor,
    "example_en": ExampleEnProcessor,
    "eprstmt_zh": EprstmtZhProcessor,
    "agnews_en": AgnewsEnProcessor
}

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET]

def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "参数 'num_examples' 和 'num_examples_per_label' 必选其一设置."

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logging.info(
        f"加载数据集 {data_dir} 配置为 ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    else:
        raise ValueError(f"'set_type' 必须是 {SET_TYPES} 中的一个, got '{set_type}' instead")

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logging.info(f"返回 {len(examples)} 条 {set_type} 样本, 标签集与文本数为: {list(label_distribution.items())}")

    return examples