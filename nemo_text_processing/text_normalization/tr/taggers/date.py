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

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_DIGIT, NEMO_SPACE, TO_LOWER, TR_LOWER, GraphFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path

# Separators accepted between the parts of a numeric date. Each gets its own branch, so
# a date must use one separator throughout and "29/09-2026" is rejected.
_NUMERIC_SEPARATORS = [".", "/", "-"]


def _numeric_range(first: int, last: int) -> "pynini.FstLike":
    """
    Maps the written forms of first..last to the bare digit string, e.g. for 1..12:
        "9" -> "9", "09" -> "9", "12" -> "12"

    A single leading zero is accepted only on one digit values, so "009", "010" and
    "0" are not in the domain. The zero is formatting and is not spoken.

    Args:
        first: smallest accepted value
        last: largest accepted value

    Returns a pynini.FstLike
    """
    pairs = [(str(n), str(n)) for n in range(first, last + 1)]
    pairs += [(f"0{n}", str(n)) for n in range(first, min(last, 9) + 1)]
    return pynini.string_map(pairs).optimize()


def _case_insensitive(words: "pynini.FstLike") -> "pynini.FstLike":
    """
    Accepts a lower case word written in lower case, capitalised or in capitals, and
    emits the lower case form, using Turkish casing, e.g.
        "nisan", "Nisan", "NİSAN" -> "nisan"

    "NISAN" is not accepted: in Turkish it lower cases to "nısan". Mixed case forms
    such as "nİSan" are not accepted either.

    Args:
        words: acceptor over lower case words

    Returns a pynini.FstLike
    """
    lower = pynini.closure(TR_LOWER, 1)
    capitalised = TO_LOWER + pynini.closure(TR_LOWER)
    capitals = pynini.closure(TO_LOWER, 2)
    return (pynini.union(lower, capitalised, capitals) @ words).optimize()


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Turkish dates, e.g.
        "29 Eylül 2026" -> date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }
        "29 Eyl. 2026" -> date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }
        "29.09.2026" -> date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }
        "29 Eylül" -> date { day: "yirmi dokuz" month: "eylül" preserve_order: true }
        "Eylül 2026" -> date { month: "eylül" year: "iki bin yirmi altı" preserve_order: true }
        "2026-09-29" -> date { year: "iki bin yirmi altı" month: "eylül" day: "yirmi dokuz" }

    Turkish reads a date day, month, year, with the day as a plain cardinal and no
    connecting words: 29 Eylül 2026 is "yirmi dokuz eylül iki bin yirmi altı". The day
    and year are read by CardinalFst; the month comes from data/dates.

    Numeric dates are day/month/year, the Turkish order, with ".", "/" or "-" used
    consistently. There is no month/day/year reading, so 03/04/2026 is 3 April. All
    three parts are required: "3/4" belongs to FractionFst, and a two part "29.09"
    would collide with ordinals and decimals.

    Year first dates, such as ISO "2026-09-29", are accepted with a four digit year in
    front, which cannot be a day. Their fields are emitted in the written order, year
    first, without ``preserve_order``, as the en grammar does for its year first dates;
    the token parser's field permutation in normalize.py then offers the verbalizer the
    day/month/year order it reads. Every other form is written in speaking order and
    carries ``preserve_order``.

    Days are 1-31 and months 1-12, either optionally with one leading zero. Day and
    month are not checked against each other, so 31/04/2026 and 29/02/2023 are
    accepted; like the en, hu and vi grammars, this validates each part's range, not
    the Gregorian calendar.

    Years are exactly four digits without a leading zero, read as a cardinal: 2026 is
    "iki bin yirmi altı". Two digit years such as 29/09/26 are not accepted, which
    avoids having to choose between reading "26" as written and inventing a century.

    Month names are accepted in lower case, capitalised or in capitals, with Turkish
    casing ("NİSAN", not "NISAN"). Abbreviations are the CLDR Turkish abbreviated
    month names (Oca, Şub, Mar, Nis, May, Haz, Tem, Ağu, Eyl, Eki, Kas, Ara), which are
    also what the CLDR medium date format uses ("29 Eyl 2026"), with or without a
    following full stop.

    Not handled here: suffixed forms such as "29 Eylül'de" and "2026'da", weekdays,
    date ranges, eras and Roman numeral months ("29.IX.2026").

    For the classifier: a slash expression is a date only with three parts, and a
    fraction only with two, so "3/4" and "03/04/2026" never compete between the two
    grammars. A sentence final full stop is never part of a date, so "29.09.2026." has
    to be split into the date and punctuation; the full stop of an abbreviation such as
    "Eyl." is part of the date. "29.09.2026" contains no three digit group, so the
    grouped cardinal cannot claim it, but a tokenizer splitting on "." could still
    offer "29." to OrdinalFst, which the date weight has to outrank.

    Args:
        cardinal: CardinalFst, supplies the readings of the day and the year
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        month_names = pynini.string_file(get_abs_path("data/dates/months.tsv"))
        month_abbr = pynini.string_file(get_abs_path("data/dates/month_abbr.tsv"))

        # "29" -> "yirmi dokuz", "09" -> "dokuz"
        self.day_graph = pynini.compose(_numeric_range(1, 31), cardinal.graph).optimize()

        # "09" -> "eylül"
        self.numeric_month_graph = pynini.compose(_numeric_range(1, 12), month_names).optimize()

        # "Eylül" -> "eylül", "Eyl." -> "eylül"
        full_month = _case_insensitive(pynini.project(month_names, "output"))
        abbr_month = _case_insensitive(pynini.project(month_abbr, "input")) @ month_abbr
        abbr_month += pynini.closure(pynutil.delete("."), 0, 1)
        self.textual_month_graph = pynini.union(full_month, abbr_month).optimize()

        self.month_graph = pynini.union(self.numeric_month_graph, self.textual_month_graph).optimize()

        # "2026" -> "iki bin yirmi altı"
        four_digit_year = (NEMO_DIGIT - "0") + NEMO_DIGIT**3
        self.year_graph = pynini.compose(four_digit_year, cardinal.graph).optimize()

        day = pynutil.insert("day: \"") + self.day_graph + pynutil.insert("\"")
        numeric_month = pynutil.insert("month: \"") + self.numeric_month_graph + pynutil.insert("\"")
        textual_month = pynutil.insert("month: \"") + self.textual_month_graph + pynutil.insert("\"")
        year = pynutil.insert("year: \"") + self.year_graph + pynutil.insert("\"")

        # 29.09.2026, 29/09/2026, 29-09-2026
        numeric_dmy = pynini.union(
            *[
                day + pynini.cross(sep, NEMO_SPACE) + numeric_month + pynini.cross(sep, NEMO_SPACE) + year
                for sep in _NUMERIC_SEPARATORS
            ]
        )

        # 29 Eylül 2026, 29 Eylül, Eylül 2026
        space = pynini.accep(NEMO_SPACE)
        textual = pynini.union(
            day + space + textual_month + space + year,
            day + space + textual_month,
            textual_month + space + year,
        )

        # 2026-09-29, 2026/09/29, 2026.09.29
        numeric_ymd = pynini.union(
            *[
                year + pynini.cross(sep, NEMO_SPACE) + numeric_month + pynini.cross(sep, NEMO_SPACE) + day
                for sep in _NUMERIC_SEPARATORS
            ]
        )

        self.final_graph = pynini.union(
            (numeric_dmy | textual) + pynutil.insert(" preserve_order: true"),
            numeric_ymd,
        ).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()
