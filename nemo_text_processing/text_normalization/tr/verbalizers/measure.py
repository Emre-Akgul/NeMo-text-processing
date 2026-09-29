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


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish measurements, e.g.
        measure { cardinal { integer: "beş" } units: "kilogram" preserve_order: true } -> beş kilogram
        measure { fraction { denominator: "dörtte" numerator: "üç" } units: "kilogram" preserve_order: true }
            -> dörtte üç kilogram

    The nested number is read by the cardinal, decimal and fraction verbalizers
    themselves, then the unit.

    Args:
        cardinal: verbalizer CardinalFst
        decimal: verbalizer DecimalFst
        fraction: verbalizer FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        number = pynini.union(cardinal.fst, decimal.fst, fraction.fst)
        units = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = number + delete_space + insert_space + units

        self.graph = graph
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
