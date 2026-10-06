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

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst

# The hyphen and the en dash, both written between the ends of a range.
_DASHES = ["-", "–"]


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying two numbers joined by a dash, a range or a
    score, e.g.
        "2-5" -> range { value: "iki beş" }
        "1995-96" -> range { value: "bin dokuz yüz doksan beş doksan altı" }
        "1995-96'da" -> range { value: "bin dokuz yüz doksan beş doksan altıda" }
        "3-4 kg" -> range { value: "üç dört kilogram" }
        "%10-15" -> range { value: "yüzde on on beş" }
        "14.00-16.00" -> range { value: "on dört on altı" }
        "3-4." -> range { value: "üçüncü dördüncü" }

    The dash is not read: both ends are read one after the other, as Turkish speakers
    read an approximate count ("3-4 dakika" -> "üç dört dakika") and a score ("2-1"
    -> "iki bir"), which a reading such as "ila" or "-den -e" would get wrong. Each end
    is read by its own grammar, so a shortened year stays a number ("1995-96").

    The first end is a cardinal, a decimal, a percentage or a time. The second is any
    of these, a measurement, an amount of money, a date or a suffixed number, all
    written with a leading digit or percent sign: the unit, currency, month or suffix
    after the second end belongs to the whole range ("3-4 kg", "10-15 TL", "1-3
    Ekim", "1995-96'da"). When the second end is an ordinal, the first is read as one
    too ("3-4." -> "üçüncü dördüncü").

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        time: TimeFst
        percentage: PercentageFst
        money: MoneyFst
        measure: MeasureFst
        suffix: SuffixFst, which also supplies the decimal and date readings
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        cardinal: GraphFst,
        ordinal: GraphFst,
        time: GraphFst,
        percentage: GraphFst,
        money: GraphFst,
        measure: GraphFst,
        suffix: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        dash = pynini.cross(pynini.union(*_DASHES), " ")

        first = pynini.union(cardinal.graph, suffix.decimal_reading, percentage.graph, time.graph)
        numeric = pynini.union(NEMO_DIGIT, "%") + NEMO_SIGMA
        numeric_suffixes = [
            suffix.branches[name] for name in ["cardinal", "decimal", "date", "time", "percentage", "money", "measure"]
        ]
        second = pynini.compose(
            numeric,
            pynini.union(first, measure.graph, money.graph, suffix.date_reading, *numeric_suffixes),
        )
        ordinals = ordinal.bare_ordinals + dash + ordinal.graph

        self.graph = pynini.union(first + dash + second, ordinals).optimize()
        self.fst = self.add_tokens(pynutil.insert("value: \"") + self.graph + pynutil.insert("\"")).optimize()
