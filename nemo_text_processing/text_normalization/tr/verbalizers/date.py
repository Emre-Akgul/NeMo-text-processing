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


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish dates, e.g.
        date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }
            -> yirmi dokuz eylül iki bin yirmi altı
        date { day: "yirmi dokuz" month: "eylül" preserve_order: true } -> yirmi dokuz eylül
        date { month: "eylül" year: "iki bin yirmi altı" preserve_order: true } -> eylül iki bin yirmi altı

    The fields are read day, month, year, which is the Turkish order, with nothing
    inserted between them. Year first dates reach this verbalizer in that order through
    the token parser's field permutation in normalize.py; see the tagger's docstring.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        def field(key: str) -> "pynini.FstLike":
            return (
                pynutil.delete(f"{key}:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
            )

        day = field("day")
        month = field("month")
        year = field("year")
        separator = delete_space + insert_space

        graph = pynini.union(
            day + separator + month + separator + year,
            day + separator + month,
            month + separator + year,
        )

        self.graph = graph
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
