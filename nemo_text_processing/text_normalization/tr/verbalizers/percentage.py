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

from nemo_text_processing.text_normalization.tr.graph_utils import TR_PERCENT_WORD, GraphFst, delete_space


class PercentageFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish percentages, e.g.
        percentage { cardinal { integer: "yirmi beş" } } -> yüzde yirmi beş
        percentage { decimal { integer_part: "on iki" fractional_part: "beş" } } -> yüzde on iki virgül beş

    "yüzde" is read first, then the nested number, which is read by the cardinal and
    decimal verbalizers themselves.

    Args:
        cardinal: verbalizer CardinalFst
        decimal: verbalizer DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="percentage", kind="verbalize", deterministic=deterministic)

        def nested(name: str, numbers: "pynini.FstLike") -> "pynini.FstLike":
            return (
                pynutil.delete(name)
                + delete_space
                + pynutil.delete("{")
                + delete_space
                + numbers
                + delete_space
                + pynutil.delete("}")
            )

        number = pynini.union(nested("cardinal", cardinal.numbers), nested("decimal", decimal.numbers))
        graph = pynutil.insert(f"{TR_PERCENT_WORD} ") + number

        self.graph = graph
        self.fst = self.delete_tokens(graph).optimize()
