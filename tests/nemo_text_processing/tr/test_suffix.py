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

"""
Direct tests for the Turkish suffix morphology and the tagger and verbalizer of forms
with a suffix written after an apostrophe.

The expected suffixes are computed by an independent Python implementation of the
rules (``_oracle``), from the canonical spoken reading of each base, not from the
grammars under test.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

import nemo_text_processing.text_normalization.tr.morphology as morphology
from nemo_text_processing.text_normalization.tr.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.tr.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst
from nemo_text_processing.text_normalization.tr.taggers.suffix import SuffixFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.tr.verbalizers.suffix import SuffixFst as SuffixVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_suffix.txt'

# Suffixes TDK joins without an apostrophe, written with one.
_APOSTROPHE_WHERE_TDK_HAS_NONE = ["vb.'leri", "Alm.'dan", "No.'lu", "No.'suz", "T.C.'de", "T.C.'da", "5 m²'ye"]
_APOSTROPHE_WHERE_TDK_HAS_NONE += ["5 cm³'e", "5 dk.'da", "5 sa.'te", "5 sn.'de", "Dr.'a", "Prof.'ün", "İng.'yi"]
# A space between the base and the suffix, or a missing apostrophe where one is needed.
_DETACHED = ["Alm. dan", "T.C. de", "5 m² ye", "No. lu", "AŞde", "AŞ.de", "AŞ' de", "AŞ 'de"]
# Suffixes after the wrong pronunciation: the letter, not the expansion, or the wrong
# harmony.
_WRONG_NO_APOSTROPHE_ALLOMORPHS = ["Dr.e", "Prof.un", "Alm.den", "İng.yı", "T.C.da", "5 dk.de", "5 sa.ta"]
_WRONG_NO_APOSTROPHE_ALLOMORPHS += ["5 m²ya", "5 cm³a", "AŞ'da", "AŞ'nın", "Org.un", "Gen.a", "No.lı", "No.sız"]
# Not accepted: dotted initialisms other than T.C., the whitelist entries that take no
# productive suffix, ASCII powers, general powers, other forms of vb. and No., and a
# sentence final full stop.
_NOT_JOINED = ["T.D.K.de", "A.B.C.de", "P.K.K.ya", "bk.ta", "krş.ta", "Sn.a", "s.de", "vs.ler", "haz.ın", "çev.in"]
_NOT_JOINED += [
    "5 m2ye",
    "5 m2'ye",
    "5 cm3e",
    "6⁴ten",
    "2 sa.lik",
    "vb.ler",
    "vb.i",
    "No.ya",
    "Dr.a.",
    "T.C.de.",
    "5 m²ye.",
]
# "MÖ" and "MS" take an apostrophe and would be spelled after their letter names
# ("MS'de"), but their expansions are adverbial phrases with no attested suffixed
# reading.
_DEFERRED_WHITELIST_SUFFIXES = ["MÖ'de", "MÖ'den", "MS'de", "MS'den"]
_OTHER_DEFERRED = ["1980'lerde", "TDK'dekiler", "3/4'ü", "4/8'i", "100€'ya", "0532 123 45 67'yi"]
_OTHER_DEFERRED += ["https://example.com/foo'da", "1,5 milyon'da", "2'şer", "7,65'lik", "2026-09-29'da"]
_MALFORMED = ["tdk'den", "Tdk'den", "-5'ten", "TDK'den.", "'da", "2026'"]

# An independent statement of the rules.
_VOWELS = "aeıioöuü"
_VOICELESS = "çfhkpsşt"
_LOW = {"a": "a", "ı": "a", "o": "a", "u": "a", "e": "e", "i": "e", "ö": "e", "ü": "e"}
_HIGH = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}
# family: (harmony, after vowel, after voiced consonant, after voiceless consonant)
_FAMILIES = {
    "locative": (_LOW, "dV", "dV", "tV"),
    "ablative": (_LOW, "dVn", "dVn", "tVn"),
    "dative": (_LOW, "yV", "V", "V"),
    "accusative": (_HIGH, "yV", "V", "V"),
    "genitive": (_HIGH, "nVn", "Vn", "Vn"),
    "instrumental": (_LOW, "ylV", "lV", "lV"),
    "plural": (_LOW, "lVr", "lVr", "lVr"),
    "possessive_3sg": (_HIGH, "sV", "V", "V"),
    "ordinal": (_HIGH, "ncV", "VncV", "VncV"),
}
_CASE_FAMILIES = [family for family in _FAMILIES if family != "ordinal"]

# The general family, whose suffixes TDK gives as front vowel ("general, -li").
_GENERAL_FAMILY = ["general", "orgeneral", "korgeneral", "tümgeneral", "tuğgeneral"]


def _front_vowel_suffix(word, family):
    """The suffix of a consonant final word harmonizing as if its last vowel were "i",
    computed without the exception table under test."""
    harmony, _, after_voiced, after_voiceless = _FAMILIES[family]
    template = after_voiceless if word[-1] in _VOICELESS else after_voiced
    return template.replace("V", harmony["i"])


def _count_paths(fst):
    """Number of paths through an acyclic acceptor, without listing them."""
    assert fst.properties(pynini.ACYCLIC, True) & pynini.ACYCLIC
    # after topsort, state ids are in topological order
    fst = pynini.topsort(fst.copy())
    counts = {state: 0 for state in fst.states()}
    counts[fst.start()] = 1
    total = 0
    for state in fst.states():
        if fst.final(state) != pynini.Weight.zero(fst.weight_type()):
            total += counts[state]
        for arc in fst.arcs(state):
            counts[arc.nextstate] += counts[state]
    return total


# Every written allomorph of each family, the wrong ones included.
_ALLOMORPHS = {
    family: sorted(
        {template.replace("V", vowel) for template in templates[1:] for vowel in set(templates[0].values())}
    )
    for family, templates in _FAMILIES.items()
}


# Words whose suffixes do not follow their last vowel, from TDK's dictionary.
_HARMONY_EXCEPTIONS = dict(load_labels(get_abs_path("data/morphology/harmony_exceptions.tsv")))
_WHITELIST = dict(load_labels(get_abs_path("data/whitelist.tsv")))
_UNSUFFIXED = [row[0] for row in load_labels(get_abs_path("data/suffix/unsuffixed_abbreviations.tsv"))]
_LEXICAL_FORMS = dict(load_labels(get_abs_path("data/suffix/lexical_forms.tsv")))


def _oracle(spoken, family, numeral=True):
    """The written suffix a spoken stem selects, and the inflected reading."""
    harmony, after_vowel, after_voiced, after_voiceless = _FAMILIES[family]
    last_word = spoken.split(" ")[-1]
    if not numeral and last_word in _HARMONY_EXCEPTIONS:
        vowel = harmony[_HARMONY_EXCEPTIONS[last_word]]
    else:
        vowel = harmony[[c for c in spoken if c in _VOWELS][-1]]
    final = spoken[-1]
    template = after_vowel if final in _VOWELS else after_voiceless if final in _VOICELESS else after_voiced
    suffix = template.replace("V", vowel)
    stem = spoken
    if numeral and final not in _VOWELS and suffix[0] in _VOWELS and spoken.split(" ")[-1] == "dört":
        stem = spoken[: -len("dört")] + "dörd"
    return suffix, stem + suffix


def _build():
    cardinal = CardinalFst()
    decimal = DecimalFst(cardinal=cardinal)
    fraction = FractionFst(cardinal=cardinal)
    whitelist = WhiteListFst()
    grammars = {
        "cardinal": cardinal,
        "decimal": decimal,
        "date": DateFst(cardinal=cardinal),
        "time": TimeFst(cardinal=cardinal),
        "percentage": PercentageFst(cardinal=cardinal, decimal=decimal),
        "money": MoneyFst(cardinal=cardinal, decimal=decimal),
        "measure": MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction),
        "abbreviation": AbbreviationFst(whitelist=whitelist),
        "electronic": ElectronicFst(),
        "whitelist": whitelist,
    }
    others = dict(grammars, fraction=fraction, whitelist=whitelist)
    others["ordinal"] = OrdinalFst(cardinal=cardinal)
    others["telephone"] = TelephoneFst(cardinal=cardinal)
    return grammars, others


_GRAMMARS, _OTHERS = _build()


class TestSuffixMorphology:
    """The reusable morphology, independent of any written base."""

    _STEMS = ["altı", "iki", "on", "beş", "dört", "kırk", "üç", "yüz", "yedi", "dokuz", "otuz", "bin", "yüz dört"]

    @parameterized.expand([(family,) for family in _FAMILIES])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_inflection(self, family):
        spec = getattr(morphology, family.upper())
        graph = morphology.inflect(spec, morphology.NUMERAL_STEM_ALTERNATION)
        for stem in self._STEMS:
            assert rewrite.rewrites(stem, graph) == [_oracle(stem, family)[1]], stem

    @parameterized.expand([(family,) for family in _FAMILIES])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_exactly_one_written_allomorph_is_accepted(self, family):
        spec = getattr(morphology, family.upper())
        validator = morphology.suffix_validator([spec], morphology.NUMERAL_STEM_ALTERNATION)
        for stem in self._STEMS:
            suffix, inflected = _oracle(stem, family)
            accepted = []
            for allomorph in _ALLOMORPHS[family]:
                try:
                    accepted.append((allomorph, rewrite.rewrites(f"{stem}'{allomorph}", validator)))
                except rewrite.Error:
                    pass
            assert accepted == [(suffix, [inflected])], (stem, accepted)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_named_suffix_graphs(self):
        assert rewrite.rewrites("beş", morphology.ABLATIVE_SUFFIX) == ["beşten"]
        assert rewrite.rewrites("on", morphology.DATIVE_SUFFIX) == ["ona"]
        assert rewrite.rewrites("on", morphology.ACCUSATIVE_SUFFIX) == ["onu"]
        assert rewrite.rewrites("altı", morphology.GENITIVE_SUFFIX) == ["altının"]
        assert rewrite.rewrites("altı", morphology.INSTRUMENTAL_SUFFIX) == ["altıyla"]
        assert rewrite.rewrites("on", morphology.PLURAL_SUFFIX) == ["onlar"]
        assert rewrite.rewrites("altı", morphology.POSSESSIVE_3SG_SUFFIX) == ["altısı"]
        # the locative and the ordinal graphs are unchanged
        assert rewrite.rewrites("dört", morphology.LOCATIVE_SUFFIX) == ["dörtte"]
        assert rewrite.rewrites("dört", morphology.ORDINAL_MORPHOLOGY) == ["dördüncü"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_softening_beyond_dort(self):
        """Only "dört" alternates, and only with a numeral policy."""
        dative = morphology.inflect(morphology.DATIVE, morphology.NUMERAL_STEM_ALTERNATION)
        assert rewrite.rewrites("kırk", dative) == ["kırka"]
        assert rewrite.rewrites("üç", dative) == ["üçe"]
        assert rewrite.rewrites("dört yüz", dative) == ["dört yüze"]
        genitive = morphology.inflect(morphology.GENITIVE)
        assert rewrite.rewrites("tübitak", genitive) == ["tübitakın"]
        assert rewrite.rewrites("dört", genitive) == ["dörtün"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_harmony_exceptions(self):
        """TDK: "saat, -ti" and "jul, -lü" take front vowel suffixes; "kilovatsaat"
        follows its final word "saat"."""
        exceptions = morphology.HARMONY_EXCEPTIONS
        locative = morphology.inflect(morphology.LOCATIVE, exceptions=exceptions)
        accusative = morphology.inflect(morphology.ACCUSATIVE, exceptions=exceptions)
        assert rewrite.rewrites("saat", locative) == ["saatte"]
        assert rewrite.rewrites("kilometre bölü saat", accusative) == ["kilometre bölü saati"]
        assert rewrite.rewrites("kilovatsaat", accusative) == ["kilovatsaati"]
        assert rewrite.rewrites("jul", accusative) == ["julü"]
        # only whole words are exceptions
        assert rewrite.rewrites("sa", locative) == ["sada"]
        assert rewrite.rewrites("altı", locative) == ["altıda"]
        # without exceptions the regular harmony applies
        assert rewrite.rewrites("saat", morphology.inflect(morphology.LOCATIVE)) == ["saatta"]

    @parameterized.expand([(word,) for word in _GENERAL_FAMILY])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_general_family_takes_front_vowel_suffixes(self, word):
        """A property of the words, not of any abbreviation: every suffix harmonizes
        front, and the back vowel spelling is rejected."""
        for family in _CASE_FAMILIES:
            spec = getattr(morphology, family.upper())
            suffix = _front_vowel_suffix(word, family)
            graph = morphology.inflect(spec, exceptions=morphology.HARMONY_EXCEPTIONS)
            assert rewrite.rewrites(word, graph) == [word + suffix], family
            validator = morphology.suffix_validator([spec], exceptions=morphology.HARMONY_EXCEPTIONS)
            back_suffix = _FAMILIES[family][2].replace("V", _FAMILIES[family][0]["a"])
            if back_suffix != suffix:
                with pytest.raises(rewrite.Error):
                    rewrite.top_rewrite(f"{word}'{back_suffix}", validator)
        assert rewrite.rewrites(
            word, morphology.inflect(morphology.DATIVE, exceptions=morphology.HARMONY_EXCEPTIONS)
        ) == [word + "e"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_anchor_differs_from_spoken_stem(self):
        """ "TL'ye" is spelled after "te le" but inflects "lira"."""
        graph = morphology.inflect_by_anchor("te le", morphology.CASE_SUFFIXES)
        assert rewrite.rewrites("yüz lira'ye", graph) == ["yüz liraya"]
        assert rewrite.rewrites("yüz lira'nin", graph) == ["yüz liranın"]
        for wrong in ["yüz lira'ya", "yüz lira'nın"]:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(wrong, graph)


class TestSuffix:

    tagger = SuffixFst(**_GRAMMARS)
    verbalizer = SuffixVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    def _accepts(self, written):
        try:
            rewrite.top_rewrite(written, self.tagger_fst)
            return True
        except rewrite.Error:
            return False

    def _normalize(self, written):
        tags = rewrite.rewrites(written, self.tagger_fst)
        assert len(tags) == 1, f"input: {written} produced {tags}"
        readings = rewrite.rewrites(tags[0], self.verbalizer_fst)
        assert len(readings) == 1, readings
        return readings[0]

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected
        assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]
        assert rewrite.top_rewrite(test_input, self.tagger_fst) == f'suffix {{ value: "{expected}" }}'
        assert "'" not in expected and "’" not in expected

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_apostrophe_forms(self, test_input, expected):
        if "'" in test_input:
            base, suffix = test_input.rsplit("'", 1)
            assert self._normalize(f"{base}’{suffix}") == expected
            wrong_forms = [f"{base}{suffix}", f"{base}''{suffix}", f"{base} '{suffix}", f"{base}' {suffix}"]
        elif test_input in _LEXICAL_FORMS:
            base, suffix = test_input.split(".", 1)
            base += "."
            wrong_forms = [f"{base}'{suffix}", f"{base} {suffix}"]
        else:
            # joined after a full stop or a superscript: never an apostrophe
            split = max(test_input.rfind(c) for c in ".²³") + 1
            base, suffix = test_input[:split], test_input[split:]
            wrong_forms = [f"{base}'{suffix}", f"{base}’{suffix}", f"{base} {suffix}", f"{base}''{suffix}"]
        for wrong in wrong_forms:
            assert not self._accepts(wrong), wrong
        assert not self._accepts(f"{test_input}.")

    @parameterized.expand(
        [(w,) for w in ["2026'de", "3'den", "6'den", "4'a", "40'e", "14.00'da", "THY'da", "PKK'ye", "BMW'da"]]
        + [(w,) for w in ["NATO'den", "TÜBİTAĞ'ın", "2'inci", "8.'inci", "100 TL'ya", "100 EUR'ya", "5 kg'den"]]
        + [(w,) for w in ["CPU'de", "GPU'ye", "%25'u", "29.09.2026'de", "09.15'de", "2'ya", "4'üncu", "40'i"]]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_wrong_allomorph_is_rejected(self, written):
        assert not self._accepts(written)

    @parameterized.expand([(w,) for w in _DETACHED + _WRONG_NO_APOSTROPHE_ALLOMORPHS + _NOT_JOINED])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_joined_suffix_forms_are_checked(self, written):
        assert not self._accepts(written)

    @parameterized.expand([(w,) for w in _APOSTROPHE_WHERE_TDK_HAS_NONE])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_apostrophe_where_tdk_has_none_is_rejected(self, written):
        assert not self._accepts(written)

    @parameterized.expand([(w,) for w in _DEFERRED_WHITELIST_SUFFIXES])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mo_and_ms_are_deferred(self, written):
        """Spelled after the letter names, but no attested suffixed reading of "milattan
        önce" and "milattan sonra"."""
        assert not self._accepts(written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_dotted_whitelist_sweep(self):
        """Every whitelist abbreviation ending in a full stop that takes productive
        suffixes, with every suffix, against the independent rules."""
        for written, spoken in _WHITELIST.items():
            if not written.endswith(".") or written in _UNSUFFIXED:
                continue
            for family in _CASE_FAMILIES:
                suffix, expected = _oracle(spoken, family, numeral=False)
                assert rewrite.rewrites(f"{written}{suffix}", self.tagger.graph) == [expected], (written, family)
                for allomorph in set(_ALLOMORPHS[family]) - {suffix}:
                    if allomorph not in {_oracle(spoken, f, numeral=False)[0] for f in _CASE_FAMILIES}:
                        assert not self._accepts(f"{written}{allomorph}"), (written, allomorph)
                assert not self._accepts(f"{written}'{suffix}")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_dotted_whitelist_entries(self):
        for written in _UNSUFFIXED:
            spoken = _WHITELIST[written]
            for family in _CASE_FAMILIES:
                assert not self._accepts(f"{written}{_oracle(spoken, family, numeral=False)[0]}"), written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_dotted_and_superscript_unit_sweep(self):
        """Every unit written with a final full stop or superscript, from MeasureFst's
        data, joined to every suffix; "m2" and other ASCII powers are not joined."""
        units = [u for u, _ in load_labels(get_abs_path("data/measure/unit.tsv")) if u[-1] in "²³"]
        units += [u for u, _ in load_labels(get_abs_path("data/measure/unit_abbreviations.tsv"))]
        measure = _GRAMMARS["measure"].graph
        for unit in units:
            for number in ["5", "12,5", "1.000"]:
                written = f"{number} {unit}"
                spoken = rewrite.top_rewrite(written, measure)
                for family in _CASE_FAMILIES:
                    suffix, expected = _oracle(spoken, family, numeral=False)
                    assert rewrite.rewrites(f"{written}{suffix}", self.tagger.graph) == [expected], (written, family)
                    assert not self._accepts(f"{written}'{suffix}")
        for unit, _ in load_labels(get_abs_path("data/measure/unit.tsv")):
            if unit[-1] in "23":
                for family in _CASE_FAMILIES:
                    spoken = rewrite.top_rewrite(f"5 {unit}", measure)
                    assert not self._accepts(f"5 {unit}{_oracle(spoken, family, numeral=False)[0]}"), unit

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_anchored_whitelist_sweep(self):
        """AŞ: the written suffix follows the letter names, the reading the expansion."""
        anchor = rewrite.top_rewrite("AŞ", _GRAMMARS["abbreviation"].initialism_graph)
        spoken = _WHITELIST["AŞ"]
        assert (anchor, spoken) == ("a şe", "anonim şirket")
        for family in _CASE_FAMILIES:
            written_suffix, _ = _oracle(anchor, family, numeral=False)
            _, expected = _oracle(spoken, family, numeral=False)
            assert rewrite.rewrites(f"AŞ'{written_suffix}", self.tagger.graph) == [expected], family
            assert rewrite.rewrites(f"AŞ’{written_suffix}", self.tagger.graph) == [expected], family
            assert not self._accepts(f"AŞ{written_suffix}")
            for allomorph in set(_ALLOMORPHS[family]) - {_oracle(anchor, f, numeral=False)[0] for f in _CASE_FAMILIES}:
                assert not self._accepts(f"AŞ'{allomorph}"), allomorph

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_lexical_forms(self):
        """Readings that are not the expansion inflected."""
        for written, spoken in _LEXICAL_FORMS.items():
            assert rewrite.rewrites(written, self.tagger.graph) == [spoken]
        assert rewrite.rewrites("No.lu", self.tagger.graph) == ["nolu"]
        assert rewrite.rewrites("vb.leri", self.tagger.graph) == ["ve benzerleri"]
        for wrong in ["numaralu", "numarasuz", "ve benzeriler"]:
            assert wrong not in {rewrite.top_rewrite(w, self.tagger.graph) for w in _LEXICAL_FORMS}

    @parameterized.expand([(w,) for w in _OTHER_DEFERRED + _MALFORMED])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_other_deferred_and_malformed_forms_are_rejected(self, written):
        assert not self._accepts(written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_money_code_selects_the_suffix_amount_takes_it(self):
        abbreviation = _GRAMMARS["abbreviation"].graph
        for code, word in [("TL", "lira"), ("TRY", "lira"), ("USD", "dolar"), ("EUR", "avro"), ("GBP", "sterlin")]:
            anchor = rewrite.top_rewrite(code, abbreviation)
            for family in ["locative", "ablative", "dative", "genitive", "instrumental"]:
                written_suffix, _ = _oracle(anchor, family, numeral=False)
                _, spoken = _oracle(f"yüz {word}", family, numeral=False)
                assert self._normalize(f"100 {code}'{written_suffix}") == spoken, (code, family)

    @parameterized.expand(
        [(w,) for w in ["PKK", "BMW", "CPU", "GPU", "NATO", "TÜBİTAK", "UNESCO", "ASELSAN", "BOTAŞ", "TÖMER"]]
        + [(w,) for w in ["İLESAM", "TİKA", "TÜBA", "TDK", "THY", "TRT", "TBMM", "KHK", "DSİ"]]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_acronym_suffixes_follow_the_reading(self, acronym):
        """The oracle works from the canonical reading, so PKK follows "ka"."""
        reading = rewrite.top_rewrite(acronym, _GRAMMARS["abbreviation"].graph)
        for family in _FAMILIES:
            if family == "ordinal":
                continue
            suffix, spoken = _oracle(reading, family, numeral=False)
            assert self._normalize(f"{acronym}'{suffix}") == spoken
            for allomorph in set(_ALLOMORPHS[family]) - {suffix}:
                assert not self._accepts(f"{acronym}'{allomorph}"), (acronym, allomorph)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_cardinal_sweep(self):
        """Every family on 0..300 and on numbers ending in dört, kırk and üç."""
        numbers = list(range(0, 301)) + [404, 1004, 2024, 10004, 440, 3003, 40000, 1234567]
        for number in numbers:
            spoken = rewrite.top_rewrite(str(number), _GRAMMARS["cardinal"].graph)
            for family in _FAMILIES:
                suffix, inflected = _oracle(spoken, family)
                assert rewrite.rewrites(f"{number}'{suffix}", self.tagger.graph) == [inflected], (number, family)
                for allomorph in set(_ALLOMORPHS[family]) - {suffix}:
                    assert not self._accepts(f"{number}'{allomorph}"), (number, allomorph)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs. The one overlap is intended and bounded: an
        abbreviation ending in a full stop followed by a suffix ("Alm.dan", "T.C.de",
        "No.lu") is also a well formed host name to ElectronicFst."""
        suffix_inputs = pynini.project(self.tagger_fst, "input").optimize()
        dotted_abbreviations = pynini.union(
            *[
                pynini.project(self.tagger.branches[k], "input")
                for k in ["dotted_whitelist", "lexical", "dotted_acronym"]
            ]
        ).optimize()
        for name, grammar in _OTHERS.items():
            shared = pynini.intersect(suffix_inputs, pynini.project(grammar.fst, "input").optimize()).optimize()
            if name != "electronic":
                assert shared.num_states() == 0, name
                continue
            assert pynini.difference(shared, dotted_abbreviations).optimize().num_states() == 0
            assert shared.properties(pynini.ACYCLIC, True) & pynini.ACYCLIC
            assert _count_paths(shared) == 86

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_general_family_is_global(self):
        """The exception applies in every branch whose last spoken word is one of these
        words: the dotted whitelist abbreviations, and a Phase 14 address alike."""
        assert rewrite.rewrites("Gen.e", self.tagger.graph) == ["generale"]
        assert rewrite.rewrites("Org.e", self.tagger.graph) == ["orgenerale"]
        assert rewrite.rewrites("Tuğg.de", self.tagger.graph) == ["tuğgeneralde"]
        assert rewrite.rewrites("x.orgeneral'e", self.tagger.graph) == ["x nokta orgenerale"]
        for wrong in ["Gen.a", "Org.a", "Org.un", "Tuğg.da", "x.orgeneral'a", "x.general'ın"]:
            assert not self._accepts(wrong), wrong

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_dotted_unit_needs_the_space(self):
        """ "5dk.da" is also a host name; the suffix is joined only in "5 dk.da"."""
        assert rewrite.rewrites("5 dk.da", self.tagger.graph) == ["beş dakikada"]
        assert not self._accepts("5dk.da")
        assert rewrite.rewrites("5m²ye", self.tagger.graph) == ["beş metrekareye"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 20000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 100, self.verbalizer_fst.num_states()
