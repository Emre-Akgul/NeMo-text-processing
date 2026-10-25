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


def roman_numeral(number: int) -> str:
    """The Roman numeral of a number below 40, e.g. 14 -> "XIV"."""
    tens, units = divmod(number, 10)
    return "X" * tens + ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"][units]


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

    A Roman numeral of two or more letters, II to XX, followed by a full stop is an
    ordinal too: "II. Dünya Savaşı" -> "ikinci Dünya Savaşı". Without the full stop it
    is read as an ordinal only with a suffix, by SuffixFst ("III'ün" -> "üçüncünün").

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

        # "II" -> "ikinci": Roman numerals of two or more letters, II to XX. A single
        # letter, "I", "V" or "X", is more often an initial.
        self.roman_ordinals = pynini.compose(
            pynini.string_map(
                [(roman_numeral(number), str(number)) for number in range(2, 21) if len(roman_numeral(number)) > 1]
            ),
            self.bare_ordinals,
        ).optimize()

        # "8." -> "sekizinci", "II." -> "ikinci"
        self.graph = (pynini.union(self.bare_ordinals, self.roman_ordinals) + pynutil.delete(".")).optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()
