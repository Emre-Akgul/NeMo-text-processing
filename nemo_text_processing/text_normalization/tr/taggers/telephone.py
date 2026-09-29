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

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst, insert_space

_PLUS_WORD = "artı"
_TURKEY_COUNTRY_CODE = "90"

# First digits of the three digit area or network code: 2, 3 and 4 are geographic,
# 5 mobile, 8 and 9 non-geographic services.
_GEOGRAPHIC_FIRST_DIGITS = "234"
_NON_GEOGRAPHIC_FIRST_DIGITS = "589"
# A geographic subscriber number starts with 2-9.
_GEOGRAPHIC_SUBSCRIBER_FIRST_DIGITS = "23456789"


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Turkish telephone numbers, e.g.
        "0532 123 45 67", "05321234567", "0 (532) 123 45 67"
            -> telephone { number_part: "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"
               preserve_order: true }
        "+90 532 123 45 67", "+905321234567"
            -> telephone { country_code: "artı doksan" number_part: "beş yüz otuz iki yüz yirmi üç kırk beş
               altmış yedi" preserve_order: true }
        "444 12 34" -> telephone { number_part: "dört yüz kırk dört on iki otuz dört" preserve_order: true }

    A Turkish number is a three digit area or network code and a seven digit subscriber
    number, read in the groups it is written and spoken in, 3, 3, 2 and 2 digits, each
    group as a number: 532 123 45 67 is "beş yüz otuz iki yüz yirmi üç kırk beş
    altmış yedi". A seven digit subscriber number is always read 3, 2, 2, however it is
    written, so "123 4567" is "yüz yirmi üç kırk beş altmış yedi". Zeros at the start
    of a group are read, "05" is "sıfır beş" and "045" is "sıfır kırk beş"; CardinalFst
    reads the rest of the group.

    Validation is structural, after the national numbering plan: the code starts with 2,
    3 or 4 (geographic), 5 (mobile), 8 or 9 (non-geographic), and a geographic
    subscriber number starts with 2-9. Whether a number is allocated is not checked.

    Accepted forms, with a space or a hyphen between groups:
        - with the trunk zero, read "sıfır": "0532 123 45 67", "0532 123 4567",
          "05321234567", "0 532 123 45 67", "0 (532) 123 45 67", "(0532) 123 45 67";
        - with the country code, read "artı doksan" and without the trunk zero:
          "+90 532 123 45 67", "+905321234567", "+90 (532) 123 45 67";
        - ten digits written in groups: "532 123 45 67";
        - a seven digit geographic or 444 number written in groups: "234 56 78",
          "444 12 34".
    Every written form of a number gives the same token. The fields are the usual
    ``country_code`` and ``number_part``, in speaking order, with ``preserve_order`` as
    in the de and ja grammars.

    Not accepted: other country codes, "0090", ungrouped ten and seven digit numbers
    ("5321234567", "2345678", "4441234") and bare short numbers ("112"), which
    CardinalFst reads as well; full stops, slashes and other separators; extensions,
    prompts such as "Tel:", and suffixed forms such as "0532 123 45 67'yi".

    For the classifier: no accepted input is accepted by another Turkish grammar.
    Ungrouped numbers and short numbers such as "112" need context to tell them from
    cardinals; "112'yi arayın" may resolve once suffixes are handled.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        non_zero_digit = NEMO_DIGIT - "0"
        zero_led = cardinal.zero + insert_space

        # "45" -> "kırk beş", "05" -> "sıfır beş", "00" -> "sıfır sıfır"
        self.two_digit_group = pynini.union(
            pynini.compose(non_zero_digit + NEMO_DIGIT, cardinal.graph),
            zero_led + pynini.union(cardinal.zero, cardinal.digit),
        ).optimize()
        # "532" -> "beş yüz otuz iki", "045" -> "sıfır kırk beş", "005" -> "sıfır sıfır beş"
        self.three_digit_group = pynini.union(
            pynini.compose(non_zero_digit + NEMO_DIGIT + NEMO_DIGIT, cardinal.graph),
            zero_led + self.two_digit_group,
        ).optimize()

        # One space or one hyphen between groups.
        separator = pynutil.delete(pynini.union(NEMO_SPACE, "-")) + insert_space
        # The last four subscriber digits, written "45 67" or "4567", are read as 45 and 67.
        last_four = pynini.union(
            self.two_digit_group + separator + self.two_digit_group,
            self.two_digit_group + insert_space + self.two_digit_group,
        )

        def starting_with(digits: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return pynini.compose(pynini.union(*digits) + pynini.closure(NEMO_DIGIT, 2, 2), graph)

        def subscriber(first_digits: str, compact: bool) -> "pynini.FstLike":
            first_group = starting_with(first_digits, self.three_digit_group)
            if compact:
                return first_group + insert_space + self.two_digit_group + insert_space + self.two_digit_group
            return first_group + separator + last_four

        # "123 45 67", "123-45-67", "123 4567" -> "yüz yirmi üç kırk beş altmış yedi"
        self.subscriber_graph = subscriber("0123456789", compact=False).optimize()

        def ten_digits(open_area: "pynini.FstLike", close_area: "pynini.FstLike", compact: bool) -> "pynini.FstLike":
            """Area code and subscriber number, for geographic and other codes."""
            numbers = []
            for area_digits, subscriber_digits in [
                (_GEOGRAPHIC_FIRST_DIGITS, _GEOGRAPHIC_SUBSCRIBER_FIRST_DIGITS),
                (_NON_GEOGRAPHIC_FIRST_DIGITS, "0123456789"),
            ]:
                area = starting_with(area_digits, self.three_digit_group)
                numbers.append(open_area + area + close_area + subscriber(subscriber_digits, compact))
            return pynini.union(*numbers)

        empty = pynini.accep("")
        paren_open = pynutil.delete("(")
        paren_close = pynutil.delete(")" + NEMO_SPACE) + insert_space

        # 532 123 45 67
        grouped = ten_digits(empty, separator, compact=False)
        compact = ten_digits(empty, insert_space, compact=True)
        parenthesised = ten_digits(paren_open, paren_close, compact=False)

        trunk = cardinal.zero + insert_space
        # 0532 123 45 67, 05321234567, 0 532 123 45 67, 0 (532) 123 45 67, (0532) 123 45 67
        self.national_number_graph = pynini.union(
            trunk + grouped,
            trunk + compact,
            trunk + pynutil.delete(NEMO_SPACE) + grouped,
            trunk + pynutil.delete(NEMO_SPACE) + parenthesised,
            paren_open + trunk + ten_digits(empty, paren_close, compact=False),
        ).optimize()

        # The number after "+90", without the separator that follows the code:
        # " 532 123 45 67", "-532-123-45-67", "5321234567", " (532) 123 45 67"
        self.international_number_graph = pynini.union(
            pynutil.delete(pynini.union(NEMO_SPACE, "-")) + grouped,
            compact,
            pynutil.delete(NEMO_SPACE) + parenthesised,
        ).optimize()
        # "+90" -> "artı doksan"
        country_code = pynini.cross("+", f"{_PLUS_WORD} ") + pynini.compose(_TURKEY_COUNTRY_CODE, cardinal.graph)

        # 532 123 45 67, and seven digit local and 444 numbers: 234 56 78, 444 12 34
        self.unprefixed_number_graph = grouped.optimize()
        self.local_number_graph = subscriber(_GEOGRAPHIC_SUBSCRIBER_FIRST_DIGITS, compact=False).optimize()

        domestic = pynini.union(self.national_number_graph, self.unprefixed_number_graph, self.local_number_graph)

        def field(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return pynutil.insert(f"{key}: \"") + graph + pynutil.insert("\"")

        self.final_graph = pynini.union(
            field("country_code", country_code) + insert_space + field("number_part", self.international_number_graph),
            field("number_part", domestic),
        )
        self.final_graph = (self.final_graph + pynutil.insert(" preserve_order: true")).optimize()

        # "+90 532 123 45 67" -> "artı doksan beş yüz otuz iki ...", the reading without
        # token fields.
        self.graph = pynini.union(country_code + insert_space + self.international_number_graph, domestic).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()
