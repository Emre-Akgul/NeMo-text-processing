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

from nemo_text_processing.text_normalization.tr.graph_utils import GraphFst
from nemo_text_processing.text_normalization.tr.morphology import ORDINAL_MORPHOLOGY


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Turkish ordinals, e.g.
        "1." -> ordinal { integer: "birinci" }
        "4." -> ordinal { integer: "dördüncü" }
        "2024." -> ordinal { integer: "iki bin yirmi dördüncü" }
        "1.000." -> ordinal { integer: "bininci" }

    Turkish marks an ordinal in writing with a full stop after the digits. The suffix is
    attached to the *last spoken word* of the cardinal and harmonizes with that word's
    final vowel, so the grammar reads the number with CardinalFst and then applies the
    productive suffix rule from tr.morphology rather than listing ordinal word forms.

    Args:
        cardinal: CardinalFst, supplies the bare number reading
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Digits to ordinal words, without the written full stop. Exposed because later
        # grammars read ordinals that are not written with a trailing stop of their own,
        # e.g. "XX. yüzyıl" or a date's day.
        self.bare_ordinals = pynini.compose(cardinal.graph, ORDINAL_MORPHOLOGY).optimize()

        self.graph = (self.bare_ordinals + pynutil.delete(".")).optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()
