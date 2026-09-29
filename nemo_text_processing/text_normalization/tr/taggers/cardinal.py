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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

# Number of three-digit groups the grammar composes, i.e. the largest magnitude
# ("kentilyon", 10^18) plus the six lower groups. This caps the grammar at 10^21 - 1.
_GROUP_COUNT = 7
_MAX_DIGITS = 3 * _GROUP_COUNT

# Magnitude keys from data/numbers/magnitudes.tsv, ordered from the largest group down.
# The units group has no magnitude word and is handled separately.
_MAGNITUDE_ORDER = ["quintillion", "quadrillion", "trillion", "billion", "million", "thousand"]


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Turkish cardinals, e.g.
        "100" -> cardinal { integer: "yüz" }
        "1000" -> cardinal { integer: "bin" }
        "-2000000" -> cardinal { negative: "true" integer: "iki milyon" }
        "1.234" -> cardinal { integer: "bin iki yüz otuz dört" }

    Turkish number words are fully compositional: a number is read as a sequence of
    three-digit groups, each followed by its magnitude word. Two multipliers of one are
    left unspoken, "yüz" (100) and "bin" (1000), so 100 is "yüz" rather than "bir yüz"
    and 1000 is "bin" rather than "bir bin". From "milyon" (10^6) upwards the multiplier
    is always spoken, so 1000000 is "bir milyon".

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        magnitudes = {key: word for key, word in load_labels(get_abs_path("data/numbers/magnitudes.tsv"))}

        self.zero = zero.optimize()
        self.digit = digit.optimize()
        self.ties = ties.optimize()

        hundred_word = magnitudes["hundred"]
        thousand_word = magnitudes["thousand"]

        digit_no_one = ((NEMO_DIGIT - "1") @ digit).optimize()
        delete_zero = pynutil.delete("0")

        # Hundreds place: "1" is silent before "yüz", so 100 is "yüz", 200 is "iki yüz".
        hundreds_place = pynini.union(
            pynini.cross("1", hundred_word),
            digit_no_one + insert_space + pynutil.insert(hundred_word),
        )

        # Exactly three digits with at least one non-zero digit. Every combination of
        # present and absent places is spelled out so that a space is inserted only
        # between two spoken words.
        self.three_digit_non_zero = pynini.union(
            hundreds_place + insert_space + ties + insert_space + digit,
            hundreds_place + insert_space + ties + delete_zero,
            hundreds_place + delete_zero + insert_space + digit,
            hundreds_place + delete_zero + delete_zero,
            delete_zero + ties + insert_space + digit,
            delete_zero + ties + delete_zero,
            delete_zero + delete_zero + digit,
        ).optimize()

        # Same block, but rejecting "001". Used for the thousands group, where a leading
        # one is not spoken.
        three_digit_non_zero_no_one = pynini.compose(
            NEMO_DIGIT**3 - pynini.accep("001"), self.three_digit_non_zero
        ).optimize()

        # Two-digit and three-digit sub-graphs, exposed for reuse by later grammars.
        self.two_digit_non_zero = pynini.union(
            ties + insert_space + digit,
            ties + delete_zero,
            delete_zero + digit,
        ).optimize()
        self.hundreds = self.three_digit_non_zero

        # Each group emits a leading space before every word it contributes; the leading
        # space of the whole number is stripped once at the end. This keeps the group
        # chain uniform instead of special casing the first spoken group.
        def magnitude_group(word: str) -> "pynini.FstLike":
            return pynini.union(
                pynutil.delete("000"),
                insert_space + self.three_digit_non_zero + insert_space + pynutil.insert(word),
            )

        thousands_group = pynini.union(
            pynutil.delete("000"),
            pynini.cross("001", NEMO_SPACE + thousand_word),
            insert_space + three_digit_non_zero_no_one + insert_space + pynutil.insert(thousand_word),
        )

        units_group = pynini.union(pynutil.delete("000"), insert_space + self.three_digit_non_zero)

        group_chain = pynini.accep("")
        for key in _MAGNITUDE_ORDER:
            group_chain += thousands_group if key == "thousand" else magnitude_group(magnitudes[key])
        group_chain += units_group

        delete_leading_space = pynini.cdrewrite(pynutil.delete(NEMO_SPACE), "[BOS]", "", NEMO_SIGMA)

        # Left pad the input with zeros to exactly _MAX_DIGITS so the fixed group chain
        # lines up with the right place values, then read it group by group.
        pad_with_zeros = pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
        non_zero_number = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)

        graph_non_zero = (
            non_zero_number @ pad_with_zeros @ (NEMO_DIGIT**_MAX_DIGITS) @ group_chain @ delete_leading_space
        ).optimize()

        number_graph = pynini.union(zero, graph_non_zero).optimize()

        # Turkish groups thousands with a full stop, e.g. "1.234.567".
        self.digit_string = self._digit_string_with_separators()
        self.graph = pynini.compose(self.digit_string, number_graph).optimize()

        # Digit by digit reading, e.g. "205" -> "iki sıfır beş". Consumed by later
        # grammars (years, telephone, decimal fractions).
        single_digit = (digit | zero).optimize()
        self.single_digits_graph = (single_digit + pynini.closure(insert_space + single_digit)).optimize()

        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_negative + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        self.fst = self.add_tokens(final_graph).optimize()

    @staticmethod
    def _digit_string_with_separators() -> "pynini.FstLike":
        """
        Accepts a bare digit string, or one whose thousands are grouped with full stops,
        and emits the bare digit string:
            "1234" -> "1234"
            "1.234.567" -> "1234567"

        Returns a pynini.FstLike
        """
        plain = pynini.closure(NEMO_DIGIT, 1)

        leading_group = pynini.closure(NEMO_DIGIT, 1, 3) - "0" - "00" - "000"
        delete_separator = pynutil.delete(".")
        grouped = leading_group + delete_separator + pynini.closure(NEMO_DIGIT**3 + delete_separator) + NEMO_DIGIT**3

        return pynini.union(plain, grouped).optimize()
