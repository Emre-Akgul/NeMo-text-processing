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
    NEMO_SPACE,
    TR_COMMA_WORD,
    TR_DECIMAL_SEPARATOR,
    TR_MINUS_WORD,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path

# TDK gives "bölü" as the reading of the "/" sign.
_PER_WORD = "bölü"


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying Turkish measurements, e.g.
        "5 kg", "5kg" -> measure { cardinal { integer: "beş" } units: "kilogram" preserve_order: true }
        "-1,5 km" -> measure { decimal { negative: "true" integer_part: "bir" fractional_part: "beş" }
            units: "kilometre" preserve_order: true }
        "3/4 kg" -> measure { fraction { numerator: "üç" denominator: "dörtte" } units: "kilogram"
            preserve_order: true }
        "90 km/sa" -> measure { cardinal { integer: "doksan" } units: "kilometre bölü saat" preserve_order: true }

    A measurement is a number followed by a unit symbol, read as the number then the
    unit's name: 5 kg is "beş kilogram". Turkish does not pluralise the unit.

    The number is a complete nested cardinal, decimal or fraction token from those
    grammars, sign included, as in the hu grammar, so every number reading, including
    grouping, quantities ("1,5 milyon km") and the denominator first fraction, is
    theirs and the verbalizer reuses theirs. ``preserve_order`` applies to the measure
    token only: the nested fraction is still permuted by normalize.py into the
    denominator first order its verbalizer reads.

    Units are listed in data/measure, symbol and spoken name, and are case sensitive:
    "m" is metre, "M" is nothing, "K" is kelvin. Alternative spellings, such as "L" and
    "l" or "m²" and "m2", are listed explicitly. The names follow TDK, whose dictionary
    has "vat", "jul" and "nevton" where TÜBİTAK UME writes "watt", "joule" and
    "newton"; prefixed names without a TDK entry ("milisaniye", "miliamper",
    "gigahertz") are the UME prefix plus the TDK base. The TDK abbreviations "sa.",
    "dk." and "sn." are accepted only as the last unit, since their full stop ends the
    unit. A unit may be divided by one other, read "bölü": "km/sa" is "kilometre bölü
    saat". The slash of a fraction belongs to the number and the slash of a unit to the
    unit, so "3/4 kg/m³" is "dörtte üç kilogram bölü metreküp".

    The unit follows the number after one space, as TDK writes it, or directly: "5 kg"
    and "5kg" give the same token.

    Not handled here: unit names written out ("5 kilogram"), products of units ("N·m"),
    powers written with "^", ranges ("5-10 kg"), dimensions ("3x4 m"), money per unit,
    "%", which is PercentageFst's, and suffixed forms such as "5 kg'dan". Pascal is
    left out: TDK has no unit sense for "paskal".

    For the suffix layer, ``graph`` gives the bare reading of cardinal and decimal
    measurements, "5 kg" -> "beş kilogram", whose final word the morphology in
    tr.morphology inflects. Fraction measurements are not in it: reading the
    denominator first means swapping two unbounded fields, which no transducer can do.

    For the classifier: a measurement always ends in a unit, so no input is shared with
    the number, date, time, percentage or money grammars; "3/4" is a fraction and
    "3/4 kg" a measurement. Single letter units ("5 A", "3 V", "2 t") can also be
    something else in running text, and the TDK dotted abbreviations absorb a following
    full stop ("5 dk.").

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        final_unit = pynini.union(unit, pynini.string_file(get_abs_path("data/measure/unit_abbreviations.tsv")))

        # "kg" -> "kilogram", "km/sa" -> "kilometre bölü saat"
        self.unit_graph = pynini.union(
            final_unit,
            unit + pynini.cross("/", f" {_PER_WORD} ") + final_unit,
        ).optimize()

        # One space between number and unit, or none.
        separator = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1) + insert_space

        number = pynini.union(cardinal.fst, decimal.fst, fraction.fst)
        units = pynutil.insert("units: \"") + self.unit_graph + pynutil.insert("\"")
        self.final_graph = (number + separator + units + pynutil.insert(" preserve_order: true")).optimize()

        # "5 kg" -> "beş kilogram", the reading without token fields, for cardinal and
        # decimal numbers. The decimal branches join the same graphs DecimalFst joins.
        quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
        decimal_reading = cardinal.graph + pynini.cross(TR_DECIMAL_SEPARATOR, f" {TR_COMMA_WORD} ") + decimal.graph
        quantity_reading = (
            pynini.union(decimal.cardinal_one_to_three_digits, decimal_reading) + pynini.accep(NEMO_SPACE) + quantities
        )
        bare_number = pynini.closure(pynini.cross("-", f"{TR_MINUS_WORD} "), 0, 1) + pynini.union(
            cardinal.graph, decimal_reading, quantity_reading
        )
        self.graph = (bare_number + separator + self.unit_graph).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()
