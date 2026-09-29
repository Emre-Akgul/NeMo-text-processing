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

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.tr.morphology import LOCATIVE_SUFFIX


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Turkish fractions, e.g.
        "3/4" -> fraction { numerator: "üç" denominator: "dörtte" }
        "1/2" -> fraction { numerator: "bir" denominator: "ikide" }
        "-3/4" -> fraction { negative: "true" numerator: "üç" denominator: "dörtte" }

    Turkish speaks the denominator first, marked with the locative suffix: 3/4 is
    "dörtte üç", literally "three in four". The suffix is productive, so the denominator
    is read by CardinalFst and then inflected by tr.morphology rather than listed per
    value. The inflected form is what goes in the denominator field, following the
    Hungarian grammar, which likewise stores the inflected denominator.

    The fields stay in the written order, numerator then denominator, and the token
    carries no ``preserve_order``. Turkish speaks them the other way round, and swapping
    two unbounded fields is not something a transducer can do, so the reordering is left
    to the token parser's field permutation in normalize.py, which is what that
    machinery is for. The en money grammar emits its fields already in speaking order
    instead, but it can only do that because a currency symbol is a single character it
    can carry in its state; a denominator of up to 21 digits is not.

    Numerator and denominator are bare digit strings. Grouped forms such as "1.000/2"
    are not accepted: a full stop inside a slash expression collides with dates, and the
    fraction grammar should not be the permissive one. Exactly one slash is accepted, so
    "29/09/2026" is not a fraction.

    Args:
        cardinal: CardinalFst, supplies the readings of both parts
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        digits = pynini.closure(NEMO_DIGIT, 1)
        cardinal_digits_only = pynini.compose(digits, cardinal.graph).optimize()

        # A zero denominator is not a number, so it is rejected rather than read.
        # Leading zeros are already rejected by CardinalFst, which is also what keeps
        # the "09" of a date like "29/09" from parsing as a denominator.
        non_zero_digits = pynini.difference(digits, pynini.accep("0")).optimize()

        # The numerator is an ordinary cardinal and may be zero: 0/5 is "beşte sıfır".
        self.numerator_graph = cardinal_digits_only
        self.denominator_graph = pynini.compose(
            pynini.compose(non_zero_digits, cardinal_digits_only), LOCATIVE_SUFFIX
        ).optimize()

        numerator = pynutil.insert("numerator: \"") + self.numerator_graph + pynutil.insert("\"")
        denominator = pynutil.insert("denominator: \"") + self.denominator_graph + pynutil.insert("\"")

        self.final_graph_wo_negative = (numerator + pynutil.delete("/") + insert_space + denominator).optimize()

        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.fst = self.add_tokens(optional_negative + self.final_graph_wo_negative).optimize()
