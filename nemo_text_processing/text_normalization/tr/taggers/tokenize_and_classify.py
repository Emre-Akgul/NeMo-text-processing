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
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
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
from nemo_text_processing.text_normalization.tr.taggers.punctuation import PUNCTUATION_MARKS, PunctuationFst
from nemo_text_processing.text_normalization.tr.taggers.range import RangeFst
from nemo_text_processing.text_normalization.tr.taggers.suffix import SuffixFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.tr.taggers.word import WORD_INTERNAL, WordFst
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
#   of these accept the same input, so their order only matters against splits. The
#   exception is a range, two numbers joined by a dash, which comes after the classes
#   whose own syntax includes a dash: a date or a telephone number keeps its reading.
# - Upper case abbreviations are acronyms when there is evidence for it: an entry of
#   the acronym table or of data/abbreviation/known_initialisms.tsv.
#   Any other string of capitals is weighted above an ordinary word, so "EV", "OKUL",
#   "ANKARA" stay words; the same holds for such a string with a suffix.
# - Punctuation is kept apart from the words it touches but costs more than a
#   semantic token that legitimately contains the same characters. A punctuation token
#   also costs a little per mark, so a token that can take a full stop takes it:
#   "8.," is an ordinal and a comma, "5 dk.," a measure and a comma; and a run of
#   marks stays one token, "..." rather than three.
# - Within a chunk of tokens joined by punctuation, an earlier token takes as much as it
#   can: a token after a join costs a little per character. "1,2,3" is "1,2" "," "3"
#   and "3/4/5" is "3/4" "/" "5", where the two readings would otherwise cost the
#   same. The cost is far below every other difference, so it only ever separates
#   tokenizations that would otherwise tie.
# - A word only has to cost more than any semantic token. It is kept small, 3 rather
#   than the 100 or 200 of other languages, because weights are 32 bit floats: a
#   sentence's cost grows with its words, and the tie breaker above must stay above
#   the rounding error of that sum, which it does up to about 270 words.
# - A word that mixes digits with letters or symbols ("COVID-19", "4x4", "11n'nin")
#   is not a word token. Where no semantic class reads such a run, it is kept
#   verbatim together with everything it is joined to by punctuation, so a chunk is
#   either read in full or left as written: "802.11n'nin" is not "802" read as a
#   number and ".11n'nin" kept, and "600Mbit/s" is not split at the slash. The
#   verbatim token costs more than any realistic run of semantic tokens joined by
#   punctuation, so it never replaces a reading of the whole chunk.
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
    "range": 1.09,
    "electronic": 1.1,
    "ordinal": 1.1,
    "decimal": 1.1,
    "fraction": 1.1,
    "cardinal": 1.1,
    "punctuation": 2.0,
    "punctuation_mark": 0.1,
    "joined_token_character": 0.0001,
    "word": 3,
    "verbatim": 10,
    "unattested_abbreviation": 3.01,
    "unattested_suffix": 3.01,
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
    AbbreviationFst lists it or data/abbreviation/known_initialisms.tsv does;
    otherwise a word wins. That file lists established upper case abbreviations drawn
    from TDK's examples, which are read letter by letter under the Phase 13 initialism
    policy; TDK establishes the abbreviations and their spelling, and its suffixed
    examples ("BDT'ye", "TDK'den", "THY'de", "TRT'den", "TL'nin") show suffixes chosen
    by the last letter's name. It also lists "HTTP" and "HTTPS", which ElectronicFst
    already reads letter by letter. The list is deliberately short; other
    initialisms ("PTT", "DSİ") stay words until they are added. AbbreviationFst and
    SuffixFst are unchanged; the classifier splits their languages and weights the
    parts. Lower cased input keeps the whitelist working, but cannot keep the acronym
    distinctions the lower casing has removed.

    Deterministic mode: every character sequence has a tokenization, and realistic
    input has a unique best one, whose cost is strictly below the next. The known
    exceptions are synthetic: a currency or percent sign between numbers, which either
    neighbour or the punctuation may take ("$3$", "%4$", "7,$2$$%"), and a word with
    an internal apostrophe inside a chain of numbers ("T/1/1'2L18"); such strings can
    have two tokenizations of equal cost, of which pynini's shortest path search
    reproducibly returns one. A general leftmost longest tie breaker is left
    for tokenizer hardening.

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
        self.range = RangeFst(
            cardinal=self.cardinal,
            ordinal=self.ordinal,
            time=self.time,
            percentage=self.percentage,
            money=self.money,
            measure=self.measure,
            suffix=self.suffix,
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
                ("range", self.range.fst),
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

        # A run of word characters that mixes digits with anything else: "COVID-19",
        # "4x4", "11n'nin". Where a chunk contains one, it is read only by semantic
        # classes, and what they do not read is kept verbatim: a verbatim token spans
        # word characters and the punctuation between them, so a chunk is either read
        # in full or left as written. Quotes and backslashes delimit token fields, so a
        # verbatim token never contains one.
        punctuation_mark = pynini.union(*[pynini.escape(mark) for mark in PUNCTUATION_MARKS])
        word_char = pynini.difference(NEMO_NOT_SPACE, punctuation_mark)
        in_word = pynini.union(word_char, *[pynini.escape(mark) for mark in WORD_INTERNAL])
        non_digit = pynini.difference(word_char, NEMO_DIGIT)
        mixed_run = pynini.union(
            non_digit + pynini.closure(in_word) + NEMO_DIGIT, NEMO_DIGIT + pynini.closure(in_word) + non_digit
        )
        mixed = (NEMO_SIGMA + mixed_run + NEMO_SIGMA).optimize()
        field_delimiter = pynini.union(*[pynini.escape(mark) for mark in ["\"", "\\"]])
        undelimited = pynini.closure(pynini.difference(NEMO_CHAR, field_delimiter)).optimize()
        verbatim = pynini.intersect(
            pynini.union(
                word_char, word_char + pynini.closure(pynini.difference(NEMO_NOT_SPACE, field_delimiter)) + word_char
            ),
            mixed,
        )
        semantic = [
            name for name, weight in _WEIGHTS.items() if name in self.token_graphs and weight < _WEIGHTS["word"]
        ]
        classify_mixed = pynini.union(
            *[pynutil.add_weight(self.token_graphs[name], _WEIGHTS[name]) for name in semantic],
            pynutil.add_weight(pynutil.insert("name: \"") + verbatim + pynutil.insert("\""), _WEIGHTS["verbatim"]),
        )

        per_mark = pynini.closure(pynutil.add_weight(NEMO_CHAR, _WEIGHTS["punctuation_mark"]))
        weighted_punctuation = pynini.compose(per_mark, self.punctuation.fst)
        punct = pynutil.insert("tokens { ") + pynutil.add_weight(weighted_punctuation, _WEIGHTS["punctuation"])
        punct += pynutil.insert(" }")

        # Whitespace separates chunks and becomes one space. Within a chunk nothing is
        # separated by whitespace: tokens are joined by punctuation, and punctuation
        # may stand at either end, "(5 kg),", "29.09" -> "29" "." "09", or alone. Two
        # tokens never touch without punctuation between them, so a word is never
        # split, and whitespace appears only between chunks, so tokenizations can only
        # differ within a chunk. Every character is a word character or punctuation,
        # so every sentence has a tokenization: a stretch with a mixed run has the
        # verbatim token, and quotes and backslashes are always punctuation.
        whitespace = pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
        join = pynutil.insert(" ")
        punct_run = punct + pynini.closure(join + punct)
        per_character = pynini.closure(pynutil.add_weight(NEMO_CHAR, _WEIGHTS["joined_token_character"]))

        def chain(classes: "pynini.FstLike") -> "pynini.FstLike":
            """Tokens of the given classes joined by punctuation."""
            token = pynutil.insert("tokens { ") + classes + pynutil.insert(" }")
            return token + pynini.closure(join + punct_run + join + pynini.compose(per_character, token))

        # The tokens between the punctuation at the ends of a chunk, in stretches
        # separated by punctuation with a quote or a backslash. A stretch with a mixed
        # run takes the restricted classes, any other the full set.
        stretch = pynini.union(
            pynini.compose(pynini.difference(undelimited, mixed), chain(classify)),
            pynini.compose(pynini.intersect(undelimited, mixed), chain(classify_mixed)),
        )
        delimiting_run = pynini.compose(NEMO_SIGMA + field_delimiter + NEMO_SIGMA, punct_run)
        joined_stretch = pynini.compose(per_character, stretch)
        tokens = stretch + pynini.closure(join + delimiting_run + join + joined_stretch)
        chunk = pynini.union(
            pynini.closure(punct_run + join, 0, 1) + tokens + pynini.closure(join + punct_run, 0, 1),
            punct_run,
        )
        graph = delete_space + chunk + pynini.closure(whitespace + chunk) + delete_space

        self.classify = classify
        self.fst = graph.optimize()
