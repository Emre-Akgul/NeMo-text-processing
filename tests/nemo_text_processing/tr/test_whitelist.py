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
Direct tests for the Turkish whitelist tagger.

The whitelist emits the spoken form directly in a ``name`` field, as the de and hu
whitelists do; there is no whitelist verbalizer. Spaces in a multi word reading are
non-breaking in the field, as ``convert_space`` makes them, and ordinary in the bare
``graph``.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.graph_utils import NEMO_NON_BREAKING_SPACE
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.electronic import ElectronicFst as ElectronicTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst as MeasureTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst as TelephoneTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst as DecimalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.fraction import FractionFst as FractionVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.measure import MeasureFst as MeasureVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_whitelist.txt'
_ENTRIES = load_labels(get_abs_path("data/whitelist.tsv"))
_WRITTEN = {written for written, _ in _ENTRIES}

# Read as letters or as a word: the acronym grammar's, not the whitelist's.
_ACRONYMS = ["TDK", "TBMM", "THY", "TRT", "ABD", "BM", "TV", "NATO", "ASELSAN", "BOTAŞ", "TÜBİTAK", "UNESCO"]
_ACRONYMS += ["CPU", "GPU", "AI", "T.C.", "T.", "A.B.C."]

# Abbreviations owned by other grammars, or left out because their reading is not
# fixed.
_NOT_WHITELISTED = ["sn.", "sn", "sa.", "dk.", "kg", "km", "m", "L", "mL", "Hz", "kW", "TL", "TRY", "USD", "EUR"]
_NOT_WHITELISTED += ["Oca.", "Eyl.", "www", "http", "https", "com", "vd.", "Mah.", "Cad.", "Sok.", "Hz.", "Müh."]
_NOT_WHITELISTED += ["Bşk.", "Apt.", "Bul.", "Prof. Dr.", "A.Ş.", "M.Ö.", "M.S."]

_OTHER_CLASSES = ["29.09.2026", "14.30", "3/4", "%25", "100€", "5 kg", "5 sn.", "05321234567"]
_OTHER_CLASSES += ["example.com", "emre@example.com", "123", "12,5", "14."]

# Suffixed forms: the suffix layer's.
_SUFFIXED = ["Dr.'a", "Prof.'un", "No.'lu", "No.lu", "Alm.dan", "İng.yi", "vb.leri", "Dr.a", "MÖ'de", "AŞ'nin"]


def _tr_upper(text):
    return "".join({"i": "İ", "ı": "I"}.get(c, c.upper()) for c in text)


def _tr_lower(text):
    return "".join({"İ": "i", "I": "ı"}.get(c, c.lower()) for c in text)


class TestWhiteList:

    whitelist = WhiteListFst()
    fst = whitelist.fst

    def _tag(self, written):
        tags = rewrite.rewrites(written, self.fst)
        assert len(tags) == 1, f"input: {written} produced {tags}"
        return tags[0]

    def _accepts(self, fst, written):
        try:
            rewrite.top_rewrite(written, fst)
            return True
        except rewrite.Error:
            return False

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert rewrite.rewrites(test_input, self.whitelist.graph) == [expected]

    @parameterized.expand(_ENTRIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_entry(self, written, spoken):
        """One tag, in the name field, with the TSV reading; the bare graph agrees."""
        assert self._tag(written) == f'name: "{spoken.replace(" ", NEMO_NON_BREAKING_SPACE)}"'
        assert rewrite.rewrites(written, self.whitelist.graph) == [spoken]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_data_is_consistent(self):
        with open(get_abs_path("data/whitelist.tsv"), encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        assert all(len(row) == 2 for row in rows), rows
        written = [row[0] for row in rows]
        assert len(written) == len(set(written)), "duplicate written forms"
        for w, s in rows:
            assert w == w.strip() and s == s.strip() and w and s, (w, s)
            assert "  " not in s, s
            assert s == s.replace(" ", " "), s

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_readings(self):
        graph = self.whitelist.graph
        for written, spoken in [
            ("Dr.", "doktor"),
            ("Prof.", "profesör"),
            ("Doç.", "doçent"),
            ("Av.", "avukat"),
            ("Uzm.", "uzman"),
            ("Alb.", "albay"),
            ("Gen.", "general"),
            ("Sn.", "sayın"),
            ("bk.", "bakınız"),
            ("krş.", "karşılaştırınız"),
            ("vb.", "ve benzeri"),
            ("No.", "numara"),
            ("MÖ", "milattan önce"),
            ("MS", "milattan sonra"),
            ("AŞ", "anonim şirket"),
            ("İng.", "İngilizce"),
        ]:
            assert rewrite.rewrites(written, graph) == [spoken], written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mutations_are_not_accepted(self):
        """Dropping or doubling the full stop, changing the case, or adding suffix
        looking text gives a string that is not accepted, unless it is itself an
        entry."""
        for written, _ in _ENTRIES:
            mutations = {
                written.rstrip("."),
                written + ".",
                written + "'",
                written + "'da",
                written + "da",
                _tr_upper(written),
                _tr_lower(written),
                _tr_upper(written[0]) + written[1:],
                _tr_lower(written[0]) + written[1:],
                " " + written,
                written + " ",
            }
            for mutation in mutations - {written} - _WRITTEN:
                assert not self._accepts(self.fst, mutation), f"{written} -> {mutation}"

    @parameterized.expand([("Dr.",), ("Prof.",), ("bk.",), ("vb.",), ("No.",), ("Sn.",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_final_full_stop_is_required(self, written):
        assert self._accepts(self.fst, written)
        assert not self._accepts(self.fst, written[:-1])

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_case_is_exact(self):
        for written in ["dr.", "DR.", "prof.", "PROF.", "BK.", "Bk.", "no.", "NO.", "mö", "Mö", "ms", "Ms", "aş"]:
            assert not self._accepts(self.fst, written), written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_sayin_and_saniye(self):
        """ "Sn." is "sayın"; "sn." after a number is MeasureFst's "saniye"."""
        assert rewrite.rewrites("Sn.", self.whitelist.graph) == ["sayın"]
        assert not self._accepts(self.fst, "sn.")
        cardinal = CardinalTaggerFst()
        decimal = DecimalTaggerFst(cardinal=cardinal)
        fraction = FractionTaggerFst(cardinal=cardinal)
        measure = MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction).fst
        cardinal_verbalizer = CardinalVerbalizerFst()
        measure_verbalizer = MeasureVerbalizerFst(
            cardinal=cardinal_verbalizer,
            decimal=DecimalVerbalizerFst(cardinal=cardinal_verbalizer),
            fraction=FractionVerbalizerFst(),
        ).fst
        assert rewrite.top_rewrite(rewrite.top_rewrite("5 sn.", measure), measure_verbalizer) == "beş saniye"
        assert not self._accepts(measure, "5 Sn.")

    @parameterized.expand([(acronym,) for acronym in _ACRONYMS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_acronyms_are_deferred(self, acronym):
        assert not self._accepts(self.fst, acronym)

    @parameterized.expand([(written,) for written in _NOT_WHITELISTED])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_abbreviations_left_to_other_grammars_or_out(self, written):
        assert not self._accepts(self.fst, written)

    @parameterized.expand([(written,) for written in _OTHER_CLASSES + _SUFFIXED])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_other_classes_and_suffixed_forms_are_rejected(self, written):
        assert not self._accepts(self.fst, written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_lower_cased_input(self):
        """Lower cased input is matched against the entries lower cased with Turkish
        rules."""
        lower = WhiteListFst(input_case="lower_cased")
        assert rewrite.rewrites("dr.", lower.graph) == ["doktor"]
        assert rewrite.rewrites("ing.", lower.graph) == ["İngilizce"]
        assert rewrite.rewrites("mö", lower.graph) == ["milattan önce"]
        assert not self._accepts(lower.fst, "Dr.")
        assert not self._accepts(lower.fst, "İng.")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_input_file(self, tmp_path):
        """A provided file replaces the defaults in deterministic mode and is added to
        them otherwise."""
        custom = tmp_path / "custom.tsv"
        custom.write_text("Mah.\tmahallesi\nDr.\tdoktor\n", encoding="utf-8")
        replaced = WhiteListFst(input_file=str(custom))
        assert rewrite.rewrites("Mah.", replaced.graph) == ["mahallesi"]
        assert rewrite.rewrites("Dr.", replaced.graph) == ["doktor"]
        assert not self._accepts(replaced.fst, "Prof.")
        added = WhiteListFst(deterministic=False, input_file=str(custom))
        assert rewrite.rewrites("Mah.", added.graph) == ["mahallesi"]
        assert rewrite.rewrites("Prof.", added.graph) == ["profesör"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_and_non_deterministic_agree(self):
        nondeterministic = WhiteListFst(deterministic=False)
        for written, spoken in _ENTRIES:
            assert rewrite.rewrites(written, nondeterministic.graph) == [spoken]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs: the input languages do not intersect."""
        cardinal = CardinalTaggerFst()
        decimal = DecimalTaggerFst(cardinal=cardinal)
        fraction = FractionTaggerFst(cardinal=cardinal)
        others = {
            "cardinal": cardinal.fst,
            "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
            "decimal": decimal.fst,
            "fraction": fraction.fst,
            "date": DateTaggerFst(cardinal=cardinal).fst,
            "time": TimeTaggerFst(cardinal=cardinal).fst,
            "percentage": PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst,
            "money": MoneyTaggerFst(cardinal=cardinal, decimal=decimal).fst,
            "measure": MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction).fst,
            "telephone": TelephoneTaggerFst(cardinal=cardinal).fst,
            "electronic": ElectronicTaggerFst().fst,
        }
        whitelist_inputs = pynini.project(self.fst, "input").optimize()
        for name, fst in others.items():
            shared = pynini.intersect(whitelist_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.fst.num_states() < 500, self.fst.num_states()
