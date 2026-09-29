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
    TR_COMMA_WORD,
    TR_MINUS_WORD,
    GraphFst,
    delete_space,
    insert_space,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish decimals, e.g.
        decimal { integer_part: "on iki" fractional_part: "beş" } -> on iki virgül beş
        decimal { negative: "true" integer_part: "bir" fractional_part: "beş" quantity: "milyon" }
            -> eksi bir virgül beş milyon

    Args:
        cardinal: verbalizer CardinalFst, supplies the shared integer field reader
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        self.optional_sign = pynini.cross("negative: \"true\"", f"{TR_MINUS_WORD} ")
        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        self.integer = pynutil.delete("integer_part:") + cardinal.integer

        self.fractional_default = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        self.fractional = pynutil.insert(f"{TR_COMMA_WORD} ") + self.fractional_default

        self.quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        graph = self.optional_sign + (
            (self.integer + self.quantity)
            | (self.integer + delete_space + insert_space + self.fractional + self.optional_quantity)
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
