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
    TR_DECIMAL_SEPARATOR,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path


def get_quantity(decimal: "pynini.FstLike", cardinal_up_to_thousand: "pynini.FstLike") -> "pynini.FstLike":
    """
    Returns an FST that transforms a cardinal or a decimal followed by a spelled out
    quantity word into a numeral, e.g.
        2 milyon -> integer_part: "iki" quantity: "milyon"
        1,5 milyon -> integer_part: "bir" fractional_part: "beş" quantity: "milyon"

    "bin" is deliberately not a quantity: Turkish writes thousands as digits
    ("2000", "2.000"), which CardinalFst already reads.

    Args:
        decimal: decimal FST, without a sign
        cardinal_up_to_thousand: cardinal FST restricted to one to three digits

    Returns a pynini.FstLike
    """
    quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
    delete_separating_space = pynutil.delete(NEMO_SPACE)

    res = (
        pynutil.insert("integer_part: \"")
        + cardinal_up_to_thousand
        + pynutil.insert("\"")
        + delete_separating_space
        + pynutil.insert(" quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= decimal + delete_separating_space + pynutil.insert(" quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Turkish decimals, e.g.
        "12,5" -> decimal { integer_part: "on iki" fractional_part: "beş" }
        "12,05" -> decimal { integer_part: "on iki" fractional_part: "sıfır beş" }
        "-1,5" -> decimal { negative: "true" integer_part: "bir" fractional_part: "beş" }
        "1,5 milyon" -> decimal { integer_part: "bir" fractional_part: "beş" quantity: "milyon" }

    Turkish writes the decimal separator as a comma and groups thousands with a full
    stop, so "1.234,5" is a decimal while "1.234" is a grouped cardinal and is rejected
    here.

    The fractional part is read so that the written digits can always be recovered from
    the reading. Leading zeros are read one by one as "sıfır"; what follows them has no
    leading zero of its own and is read as an ordinary cardinal. That keeps the natural
    Turkish reading of "3,14" as "üç virgül on dört" while keeping "1,1", "1,10" and
    "1,100" distinct, which a plain cardinal reading of the whole fractional part or a
    reading that trimmed trailing zeros would not.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        # Only bare digit strings here: the thousands separator belongs to the integer
        # part, so "1,2.345" must not parse.
        cardinal_digits_only = pynini.compose(pynini.closure(NEMO_DIGIT, 1), cardinal.graph).optimize()

        # A digit string with no leading zero, read as an ordinary number. The mapping
        # from such a string to its reading is one to one.
        no_leading_zero = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        cardinal_no_leading_zero = pynini.compose(no_leading_zero, cardinal_digits_only).optimize()

        zero_word = pynini.cross("0", "sıfır")
        leading_zeros = zero_word + pynini.closure(insert_space + zero_word)

        # Exactly one of these three branches matches any digit string, so the
        # fractional reading stays functional.
        self.graph = pynini.union(
            cardinal_no_leading_zero,  # 14 -> on dört, 205 -> iki yüz beş
            leading_zeros,  # 0 -> sıfır, 00 -> sıfır sıfır
            leading_zeros + insert_space + cardinal_no_leading_zero,  # 05 -> sıfır beş
        ).optimize()

        if not deterministic:
            # Digit by digit is always an acceptable alternative reading.
            self.graph |= cardinal.single_digits_graph

        delete_separator = pynutil.delete(TR_DECIMAL_SEPARATOR)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal.graph + pynutil.insert("\"")

        # Turkish always writes the integer part, so unlike English there is no bare
        # ",5" form and integer_part is never absent.
        final_graph_wo_sign = self.graph_integer + delete_separator + insert_space + self.graph_fractional
        self.final_graph_wo_sign = final_graph_wo_sign

        self.cardinal_up_to_thousand = pynini.compose(
            (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 2), cardinal.graph
        ).optimize()

        self.final_graph_wo_negative = (
            final_graph_wo_sign | get_quantity(final_graph_wo_sign, self.cardinal_up_to_thousand)
        ).optimize()

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        self.fst = self.add_tokens(final_graph).optimize()
