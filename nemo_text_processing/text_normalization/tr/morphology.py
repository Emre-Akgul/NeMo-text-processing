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
than looked up per word. The one rule needed so far is four way vowel harmony for
high vowel suffixes, where the suffix vowel is determined by the *last vowel of the
stem*:

    a, ı -> ı        o, u -> u
    e, i -> i        ö, ü -> ü

and whether the stem ends in a vowel or a consonant selects between two suffix shapes.

Only what the ordinal grammar needs is implemented here. This is deliberately not a
general Turkish morphology framework: consonant assimilation, buffer consonants other
than the ones spelled into a template, and the voicing alternations that apply to
ordinary nouns are all out of scope, and the numeral lexicon does not need them.
"""

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_SIGMA,
    NEMO_SPACE,
    TR_ALPHA,
    TR_CONSONANTS,
    bos_or_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

# Stems a suffix may be attached to are spoken number words, i.e. Turkish letters and
# the spaces between the words of a compound number.
_STEM_CHAR = pynini.union(TR_ALPHA, NEMO_SPACE).optimize()

HIGH_VOWEL_HARMONY = load_labels(get_abs_path("data/morphology/vowel_harmony.tsv"))


def harmonic_suffix(after_consonant: str, after_vowel: str) -> "pynini.FstLike":
    """
    Builds a transducer that appends a vowel harmonic suffix to a stem.

    Both arguments are templates containing ``{high}`` wherever the harmonic high vowel
    belongs. ``after_consonant`` is used when the stem ends in a consonant and
    ``after_vowel`` when it ends in a vowel, e.g. for the ordinal suffix::

        harmonic_suffix(after_consonant="{high}nc{high}", after_vowel="nc{high}")

    which turns "bir" into "birinci" and "iki" into "ikinci".

    The last vowel of the stem is located without a separate scan: a branch matches a
    vowel followed only by consonants up to the end of the string, and since consonants
    exclude vowels that vowel is necessarily the last one. The branches are mutually
    exclusive, so the result stays functional.

    Args:
        after_consonant: suffix template applied to a consonant final stem
        after_vowel: suffix template applied to a vowel final stem

    Returns a pynini.FstLike mapping a stem to the suffixed stem
    """
    anything = pynini.closure(_STEM_CHAR)
    branches = []
    for vowel, high in HIGH_VOWEL_HARMONY:
        branches.append(anything + pynini.accep(vowel) + pynutil.insert(after_vowel.format(high=high)))
        branches.append(
            anything
            + pynini.accep(vowel)
            + pynini.closure(TR_CONSONANTS, 1)
            + pynutil.insert(after_consonant.format(high=high))
        )
    return pynini.union(*branches).optimize()


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
ORDINAL_SUFFIX = harmonic_suffix(after_consonant="{high}nc{high}", after_vowel="nc{high}")

# Applied before the suffix: dört -> dörd, giving dördüncü rather than *dörtüncü.
ORDINAL_STEM_ALTERNATION = stem_alternation("data/ordinal/stem_exceptions.tsv")

# Full ordinal morphology: stem alternation followed by harmonic suffixation.
ORDINAL_MORPHOLOGY = pynini.compose(ORDINAL_STEM_ALTERNATION, ORDINAL_SUFFIX).optimize()
