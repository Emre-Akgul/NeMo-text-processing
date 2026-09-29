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
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish money, e.g.
        money { integer_part: "yüz" currency_maj: "lira" preserve_order: true } -> yüz lira
        money { integer_part: "on iki" currency_maj: "avro" fractional_part: "elli" currency_min: "sent"
            preserve_order: true } -> on iki avro elli sent
        money { fractional_part: "beş" currency_min: "kuruş" preserve_order: true } -> beş kuruş
        money { integer_part: "on iki" fractional_part: "üç yüz kırk beş" currency_maj: "avro"
            preserve_order: true } -> on iki virgül üç yüz kırk beş avro
        money { negative: "true" integer_part: "bir" fractional_part: "beş" quantity: "milyon"
            currency_maj: "lira" preserve_order: true } -> eksi bir virgül beş milyon lira

    Fields are read in the order the tagger emits them, which is the speaking order.
    A decimal amount, with or without a quantity, is read by the decimal verbalizer.

    Args:
        decimal: verbalizer DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        def field(key: str) -> "pynini.FstLike":
            return (
                pynutil.delete(f"{key}:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
            )

        separator = delete_space + insert_space
        integer = decimal.integer
        major = separator + field("currency_maj")
        minor = separator + field("fractional_part") + separator + field("currency_min")

        graph = pynini.union(
            decimal.optional_sign + integer + major,
            decimal.optional_sign + integer + major + minor,
            decimal.optional_sign + field("fractional_part") + separator + field("currency_min"),
            # decimal.numbers reads its own sign
            decimal.numbers + major,
        )

        self.graph = graph
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
