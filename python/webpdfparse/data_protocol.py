from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import List, Tuple, Literal

import numpy as np
import pandas as pd

# Manually defined paragraphs
@dataclass
class Element(DataClassJsonMixin):
    type: Literal['body', 'page', 'block', 'title', 'paragraph', 'image', 'heading', 'caption', 'quote', 'line', 'word']
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    total_words: int = -1
    children: List['Element'] = field(default_factory=list)
    value: str = ''
    address: str = ''
    absolute_address: int = 0
    meta_data: dict = field(default_factory=dict)

@dataclass
class Action:
    aoi: Element
    content_delta: int
    timestamp: int


@dataclass
class PassageMetaData(DataClassJsonMixin):
    title: str
    condition: Literal['digital', 'paper']
    template: np.ndarray
    element: Element
    df: pd.DataFrame

@dataclass
class Session(DataClassJsonMixin):
    pid: int
    passages: List[PassageMetaData]


@dataclass
class ParseResults:
    element: Element
    images: List[np.ndarray]