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
Minimal Turkish suffixation helpers, shared by the grammars that attach a suffix to a
spoken number.

Turkish suffixes are productive: the shape of a suffix is computed from the stem rather
than looked up per word. Two independent decisions are modelled here.

**Vowel harmony** picks the suffix vowel from the *last vowel of the stem*. High vowel
suffixes harmonize four ways and low vowel suffixes two ways:

    high: a, ı -> ı   e, i -> i   o, u -> u   ö, ü -> ü
    low:  a, ı, o, u -> a         e, i, ö, ü -> e

**Consonant assimilation** picks the suffix's initial consonant from the *last segment
of the stem*: a suffix beginning with d takes t after a voiceless consonant
(ç f h k p s ş t) and d elsewhere, including after any vowel.

The two are orthogonal, which is why one builder covers both: a suffix template is
given per final-segment class (vowel, voiced consonant, voiceless consonant) and the
harmonic vowel is substituted into whichever template applies.

Only what the ordinal and fraction grammars need is implemented. This is deliberately
not a general Turkish morphological analyzer: buffer consonants beyond what a template
spells out, stem-final voicing of ordinary nouns, and the compounding rules are all out
of scope, and the numeral lexicon does not need them.
"""

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_SIGMA,
    NEMO_SPACE,
    TR_ALPHA,
    TR_CONSONANTS,
    TR_VOICED_CONSONANTS,
    TR_VOICELESS_CONSONANTS,
    bos_or_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

# Stems a suffix may be attached to are spoken number words, i.e. Turkish letters and
# the spaces between the words of a compound number.
_STEM_CHAR = pynini.union(TR_ALPHA, NEMO_SPACE).optimize()

HIGH_VOWEL_HARMONY = load_labels(get_abs_path("data/morphology/vowel_harmony_high.tsv"))
LOW_VOWEL_HARMONY = load_labels(get_abs_path("data/morphology/vowel_harmony_low.tsv"))


def _suffix_by_final_segment(harmony, after_vowel: str, after_voiced: str, after_voiceless: str):
    """
    Shared builder for the suffix helpers below.

    Each template contains ``{vowel}`` wherever the harmonic vowel belongs, and one of
    them is chosen by the class of the stem's final segment.

    The last vowel of the stem is located without a separate scan: a branch matches a
    vowel followed only by consonants up to the end of the string, and since consonants
    exclude vowels that vowel is necessarily the last one. Within that, the final
    consonant is either voiced or voiceless but not both. The branches are therefore
    mutually exclusive and the result stays functional.

    Args:
        harmony: list of (stem vowel, harmonic suffix vowel) pairs
        after_vowel: template for a vowel final stem
        after_voiced: template for a stem ending in a voiced consonant
        after_voiceless: template for a stem ending in a voiceless consonant

    Returns a pynini.FstLike mapping a stem to the suffixed stem
    """
    anything = pynini.closure(_STEM_CHAR)
    inner_consonants = pynini.closure(TR_CONSONANTS)
    branches = []
    for vowel, harmonic in harmony:
        stem = anything + pynini.accep(vowel)
        branches.append(stem + pynutil.insert(after_vowel.format(vowel=harmonic)))
        branches.append(
            stem + inner_consonants + TR_VOICED_CONSONANTS + pynutil.insert(after_voiced.format(vowel=harmonic))
        )
        branches.append(
            stem + inner_consonants + TR_VOICELESS_CONSONANTS + pynutil.insert(after_voiceless.format(vowel=harmonic))
        )
    return pynini.union(*branches).optimize()


def harmonic_suffix(after_consonant: str, after_vowel: str, harmony=None) -> "pynini.FstLike":
    """
    Builds a transducer that appends a vowel harmonic suffix whose initial consonant
    does not assimilate, e.g. the ordinal suffix::

        harmonic_suffix(after_consonant="{vowel}nc{vowel}", after_vowel="nc{vowel}")

    which turns "bir" into "birinci" and "iki" into "ikinci".

    Args:
        after_consonant: suffix template applied to a consonant final stem
        after_vowel: suffix template applied to a vowel final stem
        harmony: harmony table, four way high vowel harmony by default

    Returns a pynini.FstLike mapping a stem to the suffixed stem
    """
    return _suffix_by_final_segment(
        harmony if harmony is not None else HIGH_VOWEL_HARMONY,
        after_vowel=after_vowel,
        after_voiced=after_consonant,
        after_voiceless=after_consonant,
    )


def assimilating_suffix(after_vowel: str, after_voiced: str, after_voiceless: str, harmony=None) -> "pynini.FstLike":
    """
    Builds a transducer that appends a suffix whose initial consonant assimilates in
    voicing to the stem, e.g. the locative -DA::

        assimilating_suffix(after_vowel="d{vowel}", after_voiced="d{vowel}",
                            after_voiceless="t{vowel}")

    which turns "altı" into "altıda", "on" into "onda" and "üç" into "üçte".

    Args:
        after_vowel: suffix template applied to a vowel final stem
        after_voiced: suffix template applied after a voiced consonant
        after_voiceless: suffix template applied after a voiceless consonant
        harmony: harmony table, two way low vowel harmony by default

    Returns a pynini.FstLike mapping a stem to the suffixed stem
    """
    return _suffix_by_final_segment(
        harmony if harmony is not None else LOW_VOWEL_HARMONY,
        after_vowel=after_vowel,
        after_voiced=after_voiced,
        after_voiceless=after_voiceless,
    )


def stem_alternation(labels_path: str) -> "pynini.FstLike":
    """
    Builds a transducer that rewrites the final word of a stem before a suffix is
    attached, for lexical exceptions such as Turkish ``dört`` -> ``dörd`` ("dördüncü").

    The rewrite is anchored to the end of the string and to a word boundary on the left,
    so a word only alternates where the suffix will actually land: "dört yüz" is left
    alone while "yüz dört" becomes "yüz dörd".

    Args:
        labels_path: path relative to the tr grammar root of a tsv of stem -> alternant

    Returns a pynini.FstLike; the identity for stems that have no exception
    """
    alternations = pynini.string_file(get_abs_path(labels_path))
    return pynini.cdrewrite(alternations, bos_or_space, "[EOS]", NEMO_SIGMA).optimize()


# The Turkish ordinal suffix, -(X)ncX with X harmonizing over ı/i/u/ü:
#   consonant final  bir -> birinci, on -> onuncu, yüz -> yüzüncü, kırk -> kırkıncı
#   vowel final      iki -> ikinci, altı -> altıncı, yirmi -> yirminci
ORDINAL_SUFFIX = harmonic_suffix(after_consonant="{vowel}nc{vowel}", after_vowel="nc{vowel}")

# Applied before the suffix: dört -> dörd, giving dördüncü rather than *dörtüncü.
ORDINAL_STEM_ALTERNATION = stem_alternation("data/ordinal/stem_exceptions.tsv")

# Full ordinal morphology: stem alternation followed by harmonic suffixation.
ORDINAL_MORPHOLOGY = pynini.compose(ORDINAL_STEM_ALTERNATION, ORDINAL_SUFFIX).optimize()

# The Turkish locative suffix, -DA with the vowel harmonizing over a/e and the
# consonant assimilating to d/t:
#   altı -> altıda, iki -> ikide, on -> onda, yüz -> yüzde, üç -> üçte, kırk -> kırkta
# No stem alternation applies: the locative begins with a consonant, so the stem final
# devoicing that produces "dördüncü" does not fire and 1/4 is "dörtte bir".
LOCATIVE_SUFFIX = assimilating_suffix(after_vowel="d{vowel}", after_voiced="d{vowel}", after_voiceless="t{vowel}")
