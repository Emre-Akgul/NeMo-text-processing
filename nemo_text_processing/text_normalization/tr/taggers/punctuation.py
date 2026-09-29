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

import string
import sys
from unicodedata import category

import pynini
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_NOT_SPACE, NEMO_SIGMA, GraphFst

# Every Unicode punctuation mark and every ASCII punctuation character. "[" and "]"
# are escaped, since pynini reads them as symbol brackets.
PUNCTUATION_MARKS = sorted(
    {chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")} | set(string.punctuation)
)


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation, e.g.
        "," -> name: ","
        "?!" -> name: "?!"

    Punctuation is kept as it is written; it is not read aloud. Unlike the en grammar,
    no symbol is left out for a whitelist reading. Emphasis tags such as "<b>" are
    kept whole, as in the en grammar.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="punctuation", kind="classify", deterministic=deterministic)

        self.punct_marks = PUNCTUATION_MARKS
        punct = pynini.closure(pynini.union(*[pynini.escape(mark) for mark in self.punct_marks]), 1)

        emphasis = (
            pynini.accep("<")
            + (
                (pynini.closure(NEMO_NOT_SPACE - pynini.union("<", ">"), 1) + pynini.closure(pynini.accep("/"), 0, 1))
                | (pynini.accep("/") + pynini.closure(NEMO_NOT_SPACE - pynini.union("<", ">"), 1))
            )
            + pynini.accep(">")
        )
        punct = plurals._priority_union(emphasis, punct, NEMO_SIGMA)

        self.graph = punct.optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
