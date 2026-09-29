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
    TR_COMMA_WORD,
    TR_DECIMAL_SEPARATOR,
    TR_PERCENT_WORD,
    GraphFst,
)

# Turkish writes the percent sign before the number, with no space: "%25".
_PERCENT_SIGN = "%"


class PercentageFst(GraphFst):
    """
    Finite state transducer for classifying Turkish percentages, e.g.
        "%25" -> percentage { cardinal { integer: "yirmi beş" } }
        "%1.000" -> percentage { cardinal { integer: "bin" } }
        "%12,5" -> percentage { decimal { integer_part: "on iki" fractional_part: "beş" } }

    The sign is read "yüzde" and, as written, comes first: %25 is "yüzde yirmi beş".
    The value is read as written, not simplified, so %25 is not "dörtte bir"; it does
    read the same as the fraction 25/100, which is intended.

    The number is a nested cardinal or decimal token, the same nesting the measure
    grammars of other languages use for "%" as a unit. Those grammars write and speak
    the sign after the number; Turkish does both before it, which is why this is its
    own class rather than a measure unit. The nested tokens come from CardinalFst's
    graph and DecimalFst's final_graph_wo_sign, so the readings, including grouped
    integer parts, leading zero rejection and the fractional reading policy, are
    exactly those of the number grammars, and the verbalizer reuses theirs. Decimal
    quantities ("%1,5 milyon") are not accepted.

    Only the prefix sign with no space is accepted, as TDK writes it; "25%" and
    "% 25" are rejected. Signed percentages ("-%5", "%-5") have no settled written
    form and are rejected, as are permille ("‰50") and suffixed forms ("%25'i").
    For the suffix layer, ``graph`` gives the bare reading, "%25" -> "yüzde yirmi
    beş", whose final word the morphology in tr.morphology inflects.

    For the classifier: "%25" is a percentage and "25/100" a fraction, which read the
    same; "%12,5" is a percentage and "12,5" a decimal. None of the other grammars
    accept an input starting with "%". The postfix "25%" appears in text that does not
    follow Turkish orthography and is not accepted here.

    Args:
        cardinal: CardinalFst, supplies integer readings
        decimal: DecimalFst, supplies decimal readings
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="percentage", kind="classify", deterministic=deterministic)

        delete_sign = pynutil.delete(_PERCENT_SIGN)

        cardinal_token = pynutil.insert("cardinal { integer: \"") + cardinal.graph + pynutil.insert("\" }")
        decimal_token = pynutil.insert("decimal { ") + decimal.final_graph_wo_sign + pynutil.insert(" }")

        self.cardinal_graph = (delete_sign + cardinal_token).optimize()
        self.decimal_graph = (delete_sign + decimal_token).optimize()
        self.final_graph = pynini.union(self.cardinal_graph, self.decimal_graph).optimize()

        # "%12,5" -> "yüzde on iki virgül beş", the reading without token fields. The
        # decimal branch joins the same integer and fractional graphs DecimalFst uses.
        decimal_reading = cardinal.graph + pynini.cross(TR_DECIMAL_SEPARATOR, f" {TR_COMMA_WORD} ") + decimal.graph
        self.graph = (
            pynini.cross(_PERCENT_SIGN, f"{TR_PERCENT_WORD} ") + pynini.union(cardinal.graph, decimal_reading)
        ).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()
