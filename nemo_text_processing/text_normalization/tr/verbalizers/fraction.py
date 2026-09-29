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

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_NOT_QUOTE,
    TR_MINUS_WORD,
    GraphFst,
    delete_space,
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish fractions, e.g.
        fraction { denominator: "dörtte" numerator: "üç" } -> dörtte üç
        fraction { negative: "true" denominator: "ikide" numerator: "bir" } -> eksi ikide bir

    The denominator is read first, which is the Turkish order. The tagger emits the
    fields the other way round, in the order they are written, and the token parser's
    field permutation in normalize.py supplies this order; see the tagger's docstring
    for why the swap cannot happen inside either transducer.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.cross("negative: \"true\"", f"{TR_MINUS_WORD} ")
        optional_sign = pynini.closure(optional_sign + delete_space, 0, 1)

        quoted = delete_space + pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        numerator = pynutil.delete("numerator:") + quoted
        denominator = pynutil.delete("denominator:") + quoted

        graph = optional_sign + denominator + delete_space + insert_space + numerator

        self.numbers = graph
        self.fst = self.delete_tokens(graph).optimize()
