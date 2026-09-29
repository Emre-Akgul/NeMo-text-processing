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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    TR_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
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
from nemo_text_processing.text_normalization.tr.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.tr.taggers.suffix import SuffixFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.tr.taggers.word import WordFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path
from nemo_text_processing.utils.logging import logger

# Token weights: a sentence's tokenization is the one of lowest total weight, so a
# lower weight is a stronger claim.
#
# - Every semantic class is between 1 and 1.1, so one semantic token always costs less
#   than any split of the same span into two or more tokens: "5 kg" is one measure,
#   not "5" and "kg"; "8." is an ordinal, not "8" and a full stop; "Dr." is one
#   whitelist token.
# - Among the semantic classes, the exact lexical ones come first: the whitelist, then
#   the suffixed forms, which also outrank ElectronicFst on the forms both accept
#   ("Alm.dan", "T.C.de", "No.lu" are suffixed abbreviations, not bare host names).
# - Classes whose inputs contain several parts are next; single numbers last. No two
#   of these accept the same input, so their order only matters against splits.
# - Upper case abbreviations are acronyms when there is evidence for it: an entry of
#   the acronym table or one of the initialisms TDK gives as read letter by letter.
#   Any other string of capitals is weighted above an ordinary word, so "EV", "OKUL",
#   "ANKARA" stay words; the same holds for such a string with a suffix.
# - Punctuation is kept apart from the words it touches but costs more than a
#   semantic token that legitimately contains the same characters. A punctuation token
#   also costs a little per mark, so a token that can take a full stop takes it:
#   "8.," is an ordinal and a comma, "5 dk.," a measure and a comma; and a run of
#   marks stays one token, "..." rather than three.
_WEIGHTS = {
    "whitelist": 1.01,
    "suffix": 1.02,
    "money": 1.05,
    "telephone": 1.05,
    "measure": 1.05,
    "percentage": 1.05,
    "date": 1.05,
    "time": 1.05,
    "abbreviation": 1.08,
    "electronic": 1.1,
    "ordinal": 1.1,
    "decimal": 1.1,
    "fraction": 1.1,
    "cardinal": 1.1,
    "punctuation": 2.0,
    "punctuation_mark": 0.1,
    "word": 100,
    "unattested_abbreviation": 101,
    "unattested_suffix": 101,
}


class ClassifyFst(GraphFst):
    """
    Final class that composes all other Turkish classification grammars. This class can
    process an entire sentence, e.g.
        "Toplantı 29.09.2026'da başlayacak." ->
            tokens { name: "Toplantı" } tokens { suffix { value: "yirmi dokuz eylül iki bin yirmi altıda" } }
            tokens { name: "başlayacak" } tokens { name: "." }

    For deployment, this grammar will be compiled and exported to OpenFst Finite State
    Archive (FAR) File. The token weights and the reasons for them are in _WEIGHTS.

    Upper case strings have no lexicon to tell an acronym from a word written in
    capitals. An abbreviation is classified as one when the acronym table of
    AbbreviationFst lists it or data/abbreviation/known_initialisms.tsv does: TDK's
    examples of letter by letter initialisms, and "HTTP" and "HTTPS", which
    ElectronicFst reads letter by letter; otherwise a word wins. AbbreviationFst and
    SuffixFst are unchanged; the classifier splits their languages and weights the
    parts. Lower cased input keeps the whitelist working, but cannot keep the acronym
    distinctions the lower casing has removed.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str = "cased",
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_tr_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
            return

        logger.info("Creating ClassifyFst grammars. This might take some time...")
        self._build(input_case, deterministic, whitelist)

        if far_file:
            generator_main(far_file, {"tokenize_and_classify": self.fst})
            logger.info(f"ClassifyFst grammars are saved to {far_file}.")

    def _build(self, input_case: str, deterministic: bool, whitelist: str):
        self.cardinal = CardinalFst(deterministic=deterministic)
        self.ordinal = OrdinalFst(cardinal=self.cardinal, deterministic=deterministic)
        self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic)
        self.fraction = FractionFst(cardinal=self.cardinal, deterministic=deterministic)
        self.date = DateFst(cardinal=self.cardinal, deterministic=deterministic)
        self.time = TimeFst(cardinal=self.cardinal, deterministic=deterministic)
        self.percentage = PercentageFst(cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic)
        self.money = MoneyFst(cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic)
        self.measure = MeasureFst(
            cardinal=self.cardinal, decimal=self.decimal, fraction=self.fraction, deterministic=deterministic
        )
        self.telephone = TelephoneFst(cardinal=self.cardinal, deterministic=deterministic)
        self.electronic = ElectronicFst(deterministic=deterministic)
        self.whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)
        self.abbreviation = AbbreviationFst(whitelist=self.whitelist, deterministic=deterministic)
        self.suffix = SuffixFst(
            cardinal=self.cardinal,
            decimal=self.decimal,
            date=self.date,
            time=self.time,
            percentage=self.percentage,
            money=self.money,
            measure=self.measure,
            abbreviation=self.abbreviation,
            electronic=self.electronic,
            whitelist=self.whitelist,
            deterministic=deterministic,
        )
        self.word = WordFst(deterministic=deterministic)
        self.punctuation = PunctuationFst(deterministic=deterministic)

        # Acronyms with evidence: the acronym table and TDK's initialisms.
        attested = pynini.union(
            pynini.project(self.abbreviation.acronym_graph, "input"),
            pynini.string_file(get_abs_path("data/abbreviation/known_initialisms.tsv")),
        ).optimize()
        abbreviations = pynini.project(self.abbreviation.graph, "input")
        unattested = pynini.difference(abbreviations, attested).optimize()

        def abbreviation_token(graph: "pynini.FstLike") -> "pynini.FstLike":
            return self.abbreviation.add_tokens(pynutil.insert("value: \"") + graph + pynutil.insert("\""))

        def suffix_token(graph: "pynini.FstLike") -> "pynini.FstLike":
            return self.suffix.add_tokens(pynutil.insert("value: \"") + graph + pynutil.insert("\""))

        written_suffix = pynini.union("'", "’") + pynini.closure(TR_LOWER, 1)
        acronym_suffixes = self.suffix.branches["abbreviation"]
        attested_suffixes = pynini.union(
            *[branch for name, branch in self.suffix.branches.items() if name != "abbreviation"],
            pynini.compose(attested + written_suffix, acronym_suffixes),
        )
        unattested_suffixes = pynini.compose(unattested + written_suffix, acronym_suffixes)

        # Each class's token graph, by the name of its weight.
        self.token_graphs = dict(
            [
                ("whitelist", self.whitelist.fst),
                ("suffix", suffix_token(attested_suffixes)),
                ("money", self.money.fst),
                ("telephone", self.telephone.fst),
                ("measure", self.measure.fst),
                ("percentage", self.percentage.fst),
                ("date", self.date.fst),
                ("time", self.time.fst),
                ("abbreviation", abbreviation_token(pynini.compose(attested, self.abbreviation.graph))),
                ("electronic", self.electronic.fst),
                ("ordinal", self.ordinal.fst),
                ("decimal", self.decimal.fst),
                ("fraction", self.fraction.fst),
                ("cardinal", self.cardinal.fst),
                ("word", self.word.fst),
                ("unattested_abbreviation", abbreviation_token(pynini.compose(unattested, self.abbreviation.graph))),
                ("unattested_suffix", suffix_token(unattested_suffixes)),
            ]
        )
        classify = pynini.union(*[pynutil.add_weight(fst, _WEIGHTS[name]) for name, fst in self.token_graphs.items()])

        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        per_mark = pynini.closure(pynutil.add_weight(NEMO_CHAR, _WEIGHTS["punctuation_mark"]))
        weighted_punctuation = pynini.compose(per_mark, self.punctuation.fst)
        punct = pynutil.insert("tokens { ") + pynutil.add_weight(weighted_punctuation, _WEIGHTS["punctuation"])
        punct += pynutil.insert(" }")

        # Whitespace separates chunks and becomes one space. Within a chunk nothing is
        # separated by whitespace: tokens are joined by punctuation, and punctuation
        # may stand at either end, "(5 kg),", "29.09" -> "29" "." "09", or alone. Two
        # tokens never touch without punctuation between them, so a word is never
        # split, and since whitespace appears only between chunks, every sentence has
        # one tokenization of lowest weight. Every character is a word character or
        # punctuation, so every sentence has a tokenization.
        whitespace = pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
        join = pynutil.insert(" ")
        punct_run = punct + pynini.closure(join + punct)
        tokens = token + pynini.closure(join + punct_run + join + token)
        chunk = pynini.union(
            pynini.closure(punct_run + join, 0, 1) + tokens + pynini.closure(join + punct_run, 0, 1),
            punct_run,
        )
        graph = delete_space + chunk + pynini.closure(whitespace + chunk) + delete_space

        self.classify = classify
        self.fst = graph.optimize()
