from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from transformers import GPT2Tokenizer

from .util import InputExample,get_verbalization_ids


FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    PVP类实现PVP结构, 对于不同的数据集会有不同的PVP实现.
    """

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_cal_type: str = 'avg'):
        """
        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.verbalizer_cal_type = verbalizer_cal_type

        if self.wrapper.config.wrapper_type in ['mlm', 'plm']:
            self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    # 根据verbalizer构建(label_num, max_num_verbalizer_len),对应的位置存单词的token_id, 没有单词的存-1
    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """返回 LM 中定义的 mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """返回 LM 中定义的 mask token id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """返回所有 verbalizers 数组最长的一个数组的长度"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @property
    def label_list_id(self) -> List[int]:
        """获取标签集的token_id"""
        label_list = self.wrapper.config.label_list
        return [get_verbalization_ids(label, self.wrapper.tokenizer, force_single_token=True) for label in label_list]

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        给定一个 label, 返回对应的 verbalizations 数组.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0] # (batch_size, vocab_size)
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml, self.verbalizer_cal_type) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor, type) -> torch.Tensor:
        # (label_num, max_num_verbalizer_len), 对应的位置存单词的token_id, 没有单词的存-1
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # (label_num) 里面存每个verbalizer的长度
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # (label_num, filler_len[i]) 根据m2c里面存的token_id去cls_logits里面找对应logit
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        # (label_num, max_num_verbalizer_len) 改形状
        cls_logits = cls_logits * (m2c > 0).float()
        if type == 'avg':
            # (label_num) 计算每个标签映射集的平均得分
            cls_logits = cls_logits.sum(axis=1) / filler_len
        elif type == 'max':
            # (label_num) 计算每个标签映射集的最大得分
            cls_logits, max_indices = torch.max(cls_logits, dim=1)
        elif type == 'sum':
            # (label_num) 计算每个标签映射集的总和得分
            cls_logits = cls_logits.sum(axis=1)
        return cls_logits

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def encode(self, example: InputExample, labeled: bool = False) \
            -> Tuple[List[int], List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.wrapper.tokenizer
        parts_a, parts_b = self.get_parts(example)


        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        verbalizer_id = get_verbalization_ids(example.label, self.wrapper.tokenizer, force_single_token=True)

        labels = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        labels[input_ids.index(self.mask_id)] = verbalizer_id

        return input_ids, token_type_ids, labels


class ExampleZhPVP(PVP):
    VERBALIZER = {
        "好": ["好","棒"],
        "坏": ["坏","差"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, '天气:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['天气是 ', self.mask, '的:', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ExampleZhPVP.VERBALIZER[label]

class ExampleEnPVP(PVP):
    VERBALIZER = {
        # "good": ["good","sunny","beautiful","nice"],
        # "bad": ["bad","terrible","strange","rainy"]
        "good": ["good"],
        "bad": ["bad"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ' :', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, ' weather :', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['The weather is ', self.mask, ' :', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ExampleEnPVP.VERBALIZER[label]

class EprstmtZhPVP(PVP):
    VERBALIZER = {
        "0": ["好","佳","高","棒"],
        "1": ["差","糟","烂","欠"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, '态度:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['这句话的态度是 ', self.mask, '的:', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return EprstmtZhPVP.VERBALIZER[label]

class AgnewsEnPVP(PVP):
    VERBALIZER = {
        # "1": ["World", 'international', 'global'],
        # "2": ["Sports", 'basketball'],
        # "3": ["Business","management"],
        # "4": ["Tech", "technology", "science"]
        "1" : ['un', 'iraq', 'government', 'war', 'country', 'president', 'minister', 'people', 'world', 'security', 'states', 'nuclear', 'military', 'israel', 'weapons', 'afghanistan', 'killed', 'officials', 'american', 'forces', 'nation', 'political', 'death', 'iraqi', 'peace', 'international', 'human', 'rights', 'crisis', 'foreign', 'troops', 'terrorism', 'attacks', 'terrorist', 'security', 'north', 'korea', 'torture', 'diplomatic', 'weapons', 'iran', 'coalition', 'violence', 'campaign', 'conflict'],
        "2": ['team', 'game', 'season', 'play', 'win', 'coach', 'players', 'baseball', 'football', 'league', 'scored', 'teams', 'player', 'coach', 'played', 'nba', 'nhl', 'cup', 'soccer', 'basketball', 'sports', 'olympic', 'world', 'champion', 'coach', 'victory', 'injury', 'match', 'athlete', 'mvp', 'hockey', 'indoor', 'athletes', 'run', 'runners', 'team', 'inning', 'inning', 'career', 'series', 'golf', 'boxer', 'bronze', 'tournament', 'arena', 'olympics', 'athletes'],
        "3": ['company', 'million', 'market', 'stock', 'profit', 'sales', 'business', 'share', 'industry', 'economic', 'growth', 'bank', 'economy', 'investors', 'quarter', 'financial', 'investor', 'firm', 'price', 'acquisition', 'earnings', 'products', 'ceo', 'corporate', 'investing', 'revenue', 'plan', 'jobs', 'sector', 'marketplace', 'losses', 'expansion', 'global', 'investments', 'strategic', 'trading', 'operating', 'domestic', 'corporations', 'declined', 'venture', 'pay', 'profits', 'outlook', 'acquire', 'revenues', 'consumer',  'economic'],
        "4": ['research', 'computer', 'software', 'technology', 'scientists', 'system', 'study', 'new', 'users', 'data', 'internet', 'medical', 'science', 'digital', 'devices', 'information', 'health', 'technology', 'engineers', 'systems', 'network', 'cell', 'phone', 'work', 'software', 'chip', 'technology', 'computers', 'genetic', 'programs', 'engineer', 'wireless', 'clinical',  'material', 'scientific', 'device', 'technologies',  'innovation', 'technology', 'web', 'online', 'tools', 'networks', 'brain', 'genetic', 'program']

    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsEnPVP.VERBALIZER[label]

PVPS = {
    'example_zh': ExampleZhPVP,
    'example_en': ExampleEnPVP,
    'eprstmt_zh': EprstmtZhPVP,
    'agnews_en': AgnewsEnPVP
}