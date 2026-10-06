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
    DERIVATIONAL_SUFFIXES,
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
        "2007'deki" -> suffix { value: "iki bin yedideki" }
        "2000'li" -> suffix { value: "iki binli" }
        "5 kg'lık" -> suffix { value: "beş kilogramlık" }
        "1850'lerde" -> suffix { value: "bin sekiz yüz ellilerde" }

    TDK separates a suffix from a number, a lower case unit abbreviation or an upper
    case abbreviation with an apostrophe and spells it after the pronunciation: of the
    number ("1985'te"), of the unit ("kg'dan"), of the last letter ("TDK'den") or of
    the word an acronym is read as ("NATO'dan"). The written suffix is therefore
    checked, not just removed: "2026'da" is accepted and "2026'de" is not.

    The first suffix after the apostrophe is checked: locative, ablative, dative,
    accusative, genitive, instrumental, plural, third person possessive and the
    derivational -lI and -lIk ("2000'li", "7,65'lik") after every class, and the
    ordinal ("8'inci", "2'nci") after a cardinal. The morphology is in
    tr.morphology; this grammar only puts each class's spoken reading in front of it:

        written base -> spoken base, then apostrophe and suffix -> inflected reading

    Suffixes may be stacked after the first one: "2007'deki", "1850'lerde",
    "34'ünün", "%50'sini", "TDK'dekiler". Only the first suffix depends on the
    reading of the base; each later one follows the suffix before it, which is
    written as it is spoken, so the letters after the first suffix are read as
    written and not limited to the suffixes above. They are checked only for vowel
    harmony, the whole written suffix with them ("4'üncu" is not "4'üncü"), except
    for the suffixes that do not harmonize, -ki, -ken and -yor ("2016'daki").

    For a number, the suffix is chosen by the spoken reading and inflects it, and the
    one lexical stem alternation of the numerals applies before a vowel initial suffix
    ("4'e" -> "dörde", but "40'a" -> "kırka", "3'ü" -> "üçü"). Abbreviations, units,
    currencies and addresses never alternate ("TÜBİTAK'ın" -> "tübitakın"). Words
    whose suffixes do not follow their last vowel are listed in
    data/morphology/harmony_exceptions.tsv: "saat" and "jul", which TDK records, and
    "kilovatsaat", which ends in "saat" ("90 km/sa'le" -> "... saatle").

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
    Suffixes TDK joins without an apostrophe, after a full stop or a superscript, are
    accepted in that spelling, and checked the same way:
        - whitelist abbreviations ending in a full stop, after their expansion:
          "Alm.dan" -> "Almancadan", "Dr.a" -> "doktora", "yy.da" -> "yüzyılda";
          except those whose expansion is a verb, a participle, an adjective or a
          single letter, listed in data/suffix/unsuffixed_abbreviations.tsv;
        - the lexical forms in data/suffix/lexical_forms.tsv, whose reading is not the
          expansion inflected: "vb.leri" -> "ve benzerleri", and "No.lu", "No.suz",
          which TDK spells after "No." read as a word -> "nolu", "nosuz";
        - "T.C.de" -> "te cede";
        - units ending in a full stop, after a space, or in a superscript: "5 dk.da" ->
          "beş dakikada", "5 sa.te" -> "beş saatte", "5 m²ye" -> "beş metrekareye".
          "m2", written without a superscript, is not.

    The harmony exceptions are global lexical data, shared by every branch: any
    branch whose final spoken word is one of them takes the corrected suffix, the
    Phase 14 branches included.

    For the classifier: an abbreviation ending in a full stop followed by a suffix,
    "Alm.dan", "yy.da", "T.C.de", "No.lu", is also a well formed host name to
    ElectronicFst, which accepts any top level label of letters. The overlap is finite
    and intended, and is for classifier priority to resolve, not either grammar: an
    exact suffixed whitelist or acronym spelling is preferred over a bare host name,
    "Alm.dan" -> "Almancadan", "T.C.de" -> "te cede", "No.lu" -> "nolu". Explicit
    electronic syntax remains strong evidence the other way: "https://alm.dan" and
    "user@alm.dan" are addresses.

    The whitelist abbreviations in data/suffix/anchored_abbreviations.tsv ("AŞ") take
    an apostrophe and are spelled after their letter names but spoken in full, as a
    currency code is: "AŞ'de" is spelled after "a şe" and read "anonim şirkette".

    Not accepted:
        - the same forms with an apostrophe ("Alm.'dan", "5 m²'ye", "T.C.'de");
        - "MÖ" and "MS", which take an apostrophe but have no attested suffixed
          reading of their spoken expansions;
        - fractions, whose written suffix TDK ties to a "bölü" reading that
          FractionFst does not use, telephone numbers, URL paths, currency symbols,
          decimal quantities ("1,5 milyon'da"), other derivational suffixes
          ("2'şer"), and a sentence final full stop;
        - stacked suffixes where no apostrophe is written ("Alm.dakiler"): without
          the apostrophe the end of the base is not marked, and these forms overlap
          host names.

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
        whitelist: WhiteListFst
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
        whitelist: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="suffix", kind="classify", deterministic=deterministic)

        apostrophe = pynini.union(_APOSTROPHE, pynini.cross(_TYPOGRAPHIC_APOSTROPHE, _APOSTROPHE))
        written_suffix = apostrophe + pynini.intersect(pynini.closure(TR_LOWER, 1), self._harmonic())

        # Written after an apostrophe: the case suffixes and -lI, -lIk.
        apostrophe_suffixes = CASE_SUFFIXES + DERIVATIONAL_SUFFIXES
        numeral = suffix_validator(apostrophe_suffixes, NUMERAL_STEM_ALTERNATION, exceptions=HARMONY_EXCEPTIONS)
        numeral_with_ordinal = suffix_validator(
            apostrophe_suffixes + [ORDINAL], NUMERAL_STEM_ALTERNATION, exceptions=HARMONY_EXCEPTIONS
        )
        word = suffix_validator(apostrophe_suffixes, exceptions=HARMONY_EXCEPTIONS)
        # Joined without an apostrophe: the case suffixes only.
        joined_word = suffix_validator(CASE_SUFFIXES, exceptions=HARMONY_EXCEPTIONS)

        # The suffixes after the first, read as written.
        stacked = pynini.closure(TR_LOWER)

        def suffixed(base: "pynini.FstLike", validator: "pynini.FstLike") -> "pynini.FstLike":
            """written base + apostrophe + suffixes -> spoken base + apostrophe + suffixes -> inflected"""
            return pynini.compose(base + written_suffix, validator + stacked).optimize()

        # After a full stop or a superscript the suffix is written without an
        # apostrophe; one is inserted so that the same validator reads it.
        joined_suffix = pynutil.insert(_APOSTROPHE) + pynini.closure(TR_LOWER, 1)

        def joined(base: "pynini.FstLike", validator: "pynini.FstLike") -> "pynini.FstLike":
            """written base + suffix -> spoken base + apostrophe + suffix -> inflected"""
            return pynini.compose(base + joined_suffix, validator).optimize()

        ends_in_letter = NEMO_SIGMA + TR_ALPHA

        # "12,5" -> "on iki virgül beş", joining the graphs DecimalFst joins; no
        # quantities, whose last word is not a number.
        self.decimal_reading = decimal_reading = (
            cardinal.graph + pynini.cross(TR_DECIMAL_SEPARATOR, f" {TR_COMMA_WORD} ") + decimal.graph
        )
        # Dates read without field permutation, i.e. day first and textual dates.
        self.date_reading = date_reading = pynini.compose(
            date.final_graph, DateVerbalizer().graph + delete_preserve_order
        )
        # Units that end in a letter: not "m²", "m2", "°" or "dk.".
        unit_reading = pynini.compose(ends_in_letter, measure.graph)
        # Acronyms, not "T.C.".
        acronym_reading = pynini.compose(ends_in_letter, abbreviation.graph)
        # Host names and e-mail addresses, with or without a scheme, but no path.
        not_slash = pynini.difference(NEMO_CHAR, "/")
        scheme = pynini.union("http", "https", "HTTP", "HTTPS") + "://"
        address_reading = pynini.compose(pynini.closure(scheme, 0, 1) + pynini.closure(not_slash), electronic.graph)

        ends_in_full_stop = NEMO_SIGMA + "."
        # "Alm." -> "Almanca": whitelist abbreviations ending in a full stop, without
        # those listed as taking no productive suffix.
        unsuffixed = pynini.union(
            *[row[0] for row in load_labels(get_abs_path("data/suffix/unsuffixed_abbreviations.tsv"))]
        )
        dotted_whitelist = pynini.compose(
            pynini.difference(
                pynini.intersect(pynini.project(whitelist.graph, "input"), ends_in_full_stop), unsuffixed
            ),
            whitelist.graph,
        )
        # "5 dk." -> "beş dakika", "5 m²" -> "beş metrekare". A dotted unit is joined to
        # its suffix only with the space TDK writes before the unit: "5dk.da" is also a
        # well formed host name.
        dotted_unit_reading = pynini.compose(NEMO_SIGMA + NEMO_SPACE + ends_in_full_stop, measure.graph)
        superscript_unit_reading = pynini.compose(NEMO_SIGMA + pynini.union("²", "³"), measure.graph)

        self.branches = {
            "cardinal": suffixed(cardinal.graph, numeral_with_ordinal),
            "decimal": suffixed(decimal_reading, numeral),
            "date": suffixed(date_reading, numeral),
            "time": suffixed(time.graph, numeral),
            "percentage": suffixed(percentage.graph, numeral),
            "money": self._money(money, abbreviation, written_suffix) + stacked,
            "measure": suffixed(unit_reading, word),
            "abbreviation": suffixed(acronym_reading, word),
            "electronic": suffixed(address_reading, word),
            "dotted_whitelist": joined(dotted_whitelist, joined_word),
            "lexical": pynini.string_file(get_abs_path("data/suffix/lexical_forms.tsv")).optimize(),
            "dotted_acronym": joined(pynini.compose(ends_in_full_stop, abbreviation.graph), joined_word),
            "dotted_unit": joined(dotted_unit_reading, joined_word),
            "superscript_unit": joined(superscript_unit_reading, joined_word),
            "anchored_whitelist": self._anchored_whitelist(whitelist, abbreviation, written_suffix) + stacked,
        }

        # "TDK'den" -> "te de keden"
        self.graph = pynini.union(*self.branches.values()).optimize()

        self.fst = self.add_tokens(pynutil.insert("value: \"") + self.graph + pynutil.insert("\"")).optimize()

    @staticmethod
    def _harmonic() -> "pynini.FstLike":
        """
        Lower case strings whose vowels harmonize from left to right: a low vowel
        agrees with the vowel before it in backness, a high vowel also in rounding,
        and "o" and "ö", which occur in no harmonizing suffix, do not follow a vowel.
        The vowel of -ki, -ke(n) and -yo(r) may follow any vowel.
        """
        front, rounded = "eiöü", "oöuü"

        def harmonizes(before: str, after: str) -> bool:
            if after in "oö":
                return False
            same_backness = (before in front) == (after in front)
            return same_backness and (after in "ae" or (before in rounded) == (after in rounded))

        vowels = "aeıioöuü"
        consonants = pynini.closure(pynini.difference(TR_LOWER, pynini.union(*vowels)))
        disharmonic = pynini.union(
            *[before + consonants + after for before in vowels for after in vowels if not harmonizes(before, after)]
        )
        invariant = pynini.union(*vowels) + consonants + pynini.union("ki", "ke", "yo")
        violation = pynini.difference(disharmonic, invariant)
        return pynini.difference(
            pynini.closure(TR_LOWER), pynini.closure(TR_LOWER) + violation + pynini.closure(TR_LOWER)
        ).optimize()

    @staticmethod
    def _anchored_whitelist(
        whitelist: GraphFst, abbreviation: GraphFst, written_suffix: "pynini.FstLike"
    ) -> "pynini.FstLike":
        """
        Whitelist abbreviations spelled after their letter names but spoken in full:
        the suffix is checked against the letter names and appended to the
        expansion, "AŞ'de" -> "anonim şirkette".
        """
        branches = []
        for (written,) in load_labels(get_abs_path("data/suffix/anchored_abbreviations.tsv")):
            anchor = rewrite.top_rewrite(written, abbreviation.initialism_graph)
            spoken = pynini.compose(written, whitelist.graph)
            branches.append(
                pynini.compose(
                    spoken + written_suffix,
                    inflect_by_anchor(anchor, CASE_SUFFIXES + DERIVATIONAL_SUFFIXES, exceptions=HARMONY_EXCEPTIONS),
                )
            )
        return pynini.union(*branches).optimize()

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
                    written + written_suffix,
                    inflect_by_anchor(anchor, CASE_SUFFIXES + DERIVATIONAL_SUFFIXES, exceptions=HARMONY_EXCEPTIONS),
                )
            )
        return pynini.union(*branches).optimize()
