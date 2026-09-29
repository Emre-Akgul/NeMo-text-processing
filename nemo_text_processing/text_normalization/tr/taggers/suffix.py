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
from pynini.lib import pynutil, rewrite

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_CHAR,
    NEMO_SIGMA,
    NEMO_SPACE,
    TR_ALPHA,
    TR_COMMA_WORD,
    TR_DECIMAL_SEPARATOR,
    TR_LOWER,
    GraphFst,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.tr.morphology import (
    CASE_SUFFIXES,
    HARMONY_EXCEPTIONS,
    NUMERAL_STEM_ALTERNATION,
    ORDINAL,
    inflect_by_anchor,
    suffix_validator,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.tr.verbalizers.date import DateFst as DateVerbalizer

# The ASCII apostrophe and the typographic one, U+2019, which Turkish text also uses.
_APOSTROPHE = "'"
_TYPOGRAPHIC_APOSTROPHE = "’"


class SuffixFst(GraphFst):
    """
    Finite state transducer for classifying Turkish forms with a suffix written after an
    apostrophe, e.g.
        "2026'da" -> suffix { value: "iki bin yirmi altıda" }
        "4'e" -> suffix { value: "dörde" }
        "8'inci" -> suffix { value: "sekizinci" }
        "17.30'da" -> suffix { value: "on yedi otuzda" }
        "%25'i" -> suffix { value: "yüzde yirmi beşi" }
        "100 TL'ye" -> suffix { value: "yüz liraya" }
        "5 kg'dan" -> suffix { value: "beş kilogramdan" }
        "TDK'den" -> suffix { value: "te de keden" }
        "PKK'ya" -> suffix { value: "pe ka kaya" }

    TDK separates a suffix from a number, a lower case unit abbreviation or an upper
    case abbreviation with an apostrophe and spells it after the pronunciation: of the
    number ("1985'te"), of the unit ("kg'dan"), of the last letter ("TDK'den") or of
    the word an acronym is read as ("NATO'dan"). The written suffix is therefore
    checked, not just removed: "2026'da" is accepted and "2026'de" is not.

    One suffix at a time is accepted: locative, ablative, dative, accusative,
    genitive, instrumental, plural and third person possessive after every class, and
    the ordinal ("8'inci", "2'nci") after a cardinal. The morphology is in
    tr.morphology; this grammar only puts each class's spoken reading in front of it:

        written base -> spoken base, then apostrophe and suffix -> inflected reading

    For a number, the suffix is chosen by the spoken reading and inflects it, and the
    one lexical stem alternation of the numerals applies before a vowel initial suffix
    ("4'e" -> "dörde", but "40'a" -> "kırka", "3'ü" -> "üçü"). Abbreviations, units,
    currencies and addresses never alternate ("TÜBİTAK'ın" -> "tübitakın"). The few
    words whose suffixes do not follow their last vowel are listed as TDK records them,
    in data/morphology/harmony_exceptions.tsv: "90 km/sa'le" -> "... saatle".

    A currency code is the one case where the pronunciation that chooses the written
    suffix is not the one spoken: "TL'ye" is spelled after the letter names "te le",
    but "100 TL'ye" is read "yüz liraya". The suffix is checked against the code's
    reading from AbbreviationFst and appended to the spoken amount.

    Classes and forms:
        - cardinals, decimals ("12,5'te"), dates written day first or with the month
          name ("29.09.2026'da", "29 Eylül'de"), times ("14.00'te" -> "on dörtte"),
          percentages ("%4'ü" -> "yüzde dördü");
        - money with a currency code ("100 TL'ye"; symbol forms are not accepted);
        - measurements whose unit ends in a letter ("5 kg'dan", "20 °C'de"): TDK writes
          a suffix after "m²" or "cm³" without an apostrophe ("m²ye"), so those are
          not accepted here;
        - acronyms and initialisms, except "T.C.", whose final full stop takes a suffix
          without an apostrophe;
        - host names and e-mail addresses ("example.com'da"), not URLs with a path,
          whose last segment has no known Turkish pronunciation.
    Not accepted: fractions, whose written suffix TDK ties to a "bölü" reading that
    FractionFst does not use, telephone numbers, decimal quantities ("1,5 milyon'da"),
    whitelist abbreviations, which TDK suffixes without an apostrophe ("vb.leri"),
    more than one suffix ("1980'lerde"), and a sentence final full stop.

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        date: DateFst
        time: TimeFst
        percentage: PercentageFst
        money: MoneyFst
        measure: MeasureFst
        abbreviation: AbbreviationFst
        electronic: ElectronicFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        date: GraphFst,
        time: GraphFst,
        percentage: GraphFst,
        money: GraphFst,
        measure: GraphFst,
        abbreviation: GraphFst,
        electronic: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="suffix", kind="classify", deterministic=deterministic)

        apostrophe = pynini.union(_APOSTROPHE, pynini.cross(_TYPOGRAPHIC_APOSTROPHE, _APOSTROPHE))
        written_suffix = apostrophe + pynini.closure(TR_LOWER, 1)

        numeral = suffix_validator(CASE_SUFFIXES, NUMERAL_STEM_ALTERNATION, exceptions=HARMONY_EXCEPTIONS)
        numeral_with_ordinal = suffix_validator(
            CASE_SUFFIXES + [ORDINAL], NUMERAL_STEM_ALTERNATION, exceptions=HARMONY_EXCEPTIONS
        )
        word = suffix_validator(CASE_SUFFIXES, exceptions=HARMONY_EXCEPTIONS)

        def suffixed(base: "pynini.FstLike", validator: "pynini.FstLike") -> "pynini.FstLike":
            """written base + apostrophe + suffix -> spoken base + apostrophe + suffix -> inflected"""
            return pynini.compose(base + written_suffix, validator).optimize()

        ends_in_letter = NEMO_SIGMA + TR_ALPHA

        # "12,5" -> "on iki virgül beş", joining the graphs DecimalFst joins; no
        # quantities, whose last word is not a number.
        decimal_reading = cardinal.graph + pynini.cross(TR_DECIMAL_SEPARATOR, f" {TR_COMMA_WORD} ") + decimal.graph
        # Dates read without field permutation, i.e. day first and textual dates.
        date_reading = pynini.compose(date.final_graph, DateVerbalizer().graph + delete_preserve_order)
        # Units that end in a letter: not "m²", "m2", "°" or "dk.".
        unit_reading = pynini.compose(ends_in_letter, measure.graph)
        # Acronyms, not "T.C.".
        acronym_reading = pynini.compose(ends_in_letter, abbreviation.graph)
        # Host names and e-mail addresses, with or without a scheme, but no path.
        not_slash = pynini.difference(NEMO_CHAR, "/")
        scheme = pynini.union("http", "https", "HTTP", "HTTPS") + "://"
        address_reading = pynini.compose(pynini.closure(scheme, 0, 1) + pynini.closure(not_slash), electronic.graph)

        self.branches = {
            "cardinal": suffixed(cardinal.graph, numeral_with_ordinal),
            "decimal": suffixed(decimal_reading, numeral),
            "date": suffixed(date_reading, numeral),
            "time": suffixed(time.graph, numeral),
            "percentage": suffixed(percentage.graph, numeral),
            "money": self._money(money, abbreviation, written_suffix),
            "measure": suffixed(unit_reading, word),
            "abbreviation": suffixed(acronym_reading, word),
            "electronic": suffixed(address_reading, word),
        }

        # "TDK'den" -> "te de keden"
        self.graph = pynini.union(*self.branches.values()).optimize()

        self.fst = self.add_tokens(pynutil.insert("value: \"") + self.graph + pynutil.insert("\"")).optimize()

    @staticmethod
    def _money(money: GraphFst, abbreviation: GraphFst, written_suffix: "pynini.FstLike") -> "pynini.FstLike":
        """
        Money written with a currency code: the suffix is checked against the code's
        letter names and appended to the spoken amount, "100 TL'ye" -> "yüz liraya".
        """
        branches = []
        for code, _ in load_labels(get_abs_path("data/money/currency_codes.tsv")):
            written = pynini.compose(NEMO_SIGMA + NEMO_SPACE + code, money.graph)
            anchor = rewrite.top_rewrite(code, abbreviation.graph)
            branches.append(
                pynini.compose(
                    written + written_suffix, inflect_by_anchor(anchor, CASE_SUFFIXES, exceptions=HARMONY_EXCEPTIONS)
                )
            )
        return pynini.union(*branches).optimize()
