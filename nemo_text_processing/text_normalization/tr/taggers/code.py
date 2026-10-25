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
    NEMO_DIGIT,
    NEMO_SPACE,
    TO_UPPER,
    TR_ALPHA,
    TR_LOWER,
    TR_UPPER,
    GraphFst,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

# Read between the parts of a code.
_DOT_WORD = "nokta"
_TIMES_WORD = "çarpı"
_PLUS_WORD = "artı"


class CodeFst(GraphFst):
    """
    Finite state transducer for classifying alphanumeric codes, runs of letters and
    digits such as model, standard and virus names, e.g.
        "H5N1" -> code { value: "he beş ne bir" }
        "COVID-19" -> code { value: "kovid on dokuz" }
        "4x4" -> code { value: "dört çarpı dört" }
        "802.11n" -> code { value: "sekiz yüz iki nokta on bir ne" }
        "Chandrayaan-1" -> code { value: "Chandrayaan bir" }

    A code alternates runs of digits and runs of letters, which may also be separated
    by a hyphen, read as nothing, a full stop, "nokta", or a plus sign, "artı"; an "x"
    between two digit runs is "çarpı". A digit run is a cardinal, or read digit by
    digit when it starts with a zero. A run of capitals is read as AbbreviationFst
    reads it, by the acronym table or letter by letter ("COVID" -> "kovid", "GP" ->
    "ge pe"), a single letter by its name, and a run of two or more letters with a
    lower case one is a word, kept as written ("Chandrayaan"), unless it is a unit
    of MeasureFst ("600Mbit", "5-6kg" are measurements and ranges).

    A hyphen between two digit runs makes a range, which RangeFst reads, so it is not
    part of a code. A code has at least one digit and one letter; the classifier reads it only where
    a run mixes the two, which is where an ordinary word is not allowed.

    Args:
        cardinal: CardinalFst
        abbreviation: AbbreviationFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, abbreviation: GraphFst, deterministic: bool = True):
        super().__init__(name="code", kind="classify", deterministic=deterministic)

        digits = pynini.closure(NEMO_DIGIT, 1)
        zero_led = "0" + digits
        digit_run = pynini.union(
            pynini.compose(pynini.difference(digits, zero_led), cardinal.graph),
            pynini.compose(zero_led, cardinal.single_digits_graph),
        )

        # "x" between two digit runs is "çarpı", never the letter.
        single_letter = pynini.union(
            abbreviation.letter_graph,
            pynini.compose(pynini.difference(TR_LOWER, "x"), TO_UPPER @ abbreviation.letter_graph),
        )
        capitals = pynini.compose(pynini.closure(TR_UPPER, 2), abbreviation.graph)
        # A unit is not a word of a code: "600Mbit" and "5-6kg" are measurements and
        # ranges, read by those grammars.
        units = pynini.union(
            *[
                row[0]
                for file in ["data/measure/unit.tsv", "data/measure/unit_abbreviations.tsv"]
                for row in load_labels(get_abs_path(file))
            ]
        )
        word = pynini.difference(pynini.difference(pynini.closure(TR_ALPHA, 2), pynini.closure(TR_UPPER)), units)
        letter_run = pynini.union(single_letter, capitals, word)

        space = pynutil.insert(NEMO_SPACE)
        # Runs of the other kind follow each other without a separator, so every run is
        # maximal: "GP" is one run, never "G" and "P".
        alternating = pynini.union(
            digit_run
            + pynini.closure(space + letter_run + space + digit_run)
            + pynini.closure(space + letter_run, 0, 1),
            letter_run
            + pynini.closure(space + digit_run + space + letter_run)
            + pynini.closure(space + digit_run, 0, 1),
        )
        separator = pynini.union(
            pynini.cross("-", NEMO_SPACE),
            pynini.cross(".", f" {_DOT_WORD} "),
            pynini.cross("+", f" {_PLUS_WORD} "),
        )
        times = digit_run + pynini.cross("x", f" {_TIMES_WORD} ") + digit_run
        part = pynini.union(alternating, times)
        code = part + pynini.closure(separator + part)

        # At least one digit and one letter.
        anything = pynini.closure(pynini.union(NEMO_DIGIT, TR_ALPHA, "-", ".", "+"))
        mixed = pynini.intersect(anything + NEMO_DIGIT + anything, anything + TR_ALPHA + anything)
        # Two numbers joined by a hyphen are a range, "5-6kg'lık", never part of a code.
        mixed = pynini.difference(mixed, anything + NEMO_DIGIT + "-" + NEMO_DIGIT + anything)
        self.graph = pynini.compose(mixed, code).optimize()

        self.fst = self.add_tokens(pynutil.insert("value: \"") + self.graph + pynutil.insert("\"")).optimize()
