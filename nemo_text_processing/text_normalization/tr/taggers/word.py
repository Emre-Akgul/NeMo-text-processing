# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.tr.taggers.punctuation import PUNCTUATION_MARKS

# Written inside a word, between two runs of word characters: "Yılmaz'a", "e-posta".
WORD_INTERNAL = ["'", "’", "-"]


class WordFst(GraphFst):
    """
    Finite state transducer for classifying an ordinary word, kept as written, e.g.
        "merhaba" -> name: "merhaba"
        "Çayyolu" -> name: "Çayyolu"
        "Yılmaz'a" -> name: "Yılmaz'a"

    A word is a run of characters that are neither space nor punctuation; an
    apostrophe or a hyphen between two such runs belongs to the word. Punctuation at
    either end is not part of it, so "merhaba," is a word followed by punctuation.
    Nothing is lower cased or otherwise changed.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        punctuation = pynini.union(*[pynini.escape(mark) for mark in PUNCTUATION_MARKS])
        run = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punctuation), 1)
        internal = pynini.union(*[pynini.escape(mark) for mark in WORD_INTERNAL])
        self.graph = (run + pynini.closure(internal + run)).optimize()

        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
