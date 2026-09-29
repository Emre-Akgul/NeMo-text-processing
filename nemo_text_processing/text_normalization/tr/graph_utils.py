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
Turkish-specific alphabet and casing helpers.

Generic WFST infrastructure (``GraphFst``, the ``NEMO_*`` character classes and the
space helpers) is imported from the shared English implementation and re-exported here,
so Turkish grammars only need to import from this module.

Turkish casing is *not* ASCII casing: the language distinguishes a dotted and a dotless
``i``, and the case pairs are ``I`` <-> ``ı`` and ``İ`` <-> ``i``. ``TO_LOWER``/``TO_UPPER``
below implement that mapping, which ``str.lower()``/``str.upper()`` and
``en.graph_utils.TO_LOWER`` both get wrong.
"""

import pynini
from pynini.lib import byte

from nemo_text_processing.text_normalization.en.graph_utils import (  # noqa: F401
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_NOT_QUOTE,
    NEMO_NOT_SPACE,
    NEMO_PUNCT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    generator_main,
    insert_space,
)

# Turkish alphabet, upper and lower case aligned position by position.
# Note the ordering around index 10-11: "I" pairs with "ı" and "İ" pairs with "i".
_TR_ALPHA_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
_TR_ALPHA_LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"

# q, w and x are not part of the Turkish alphabet but occur in loanwords, names and
# abbreviations, so they are accepted and cased in the ordinary ASCII way.
_FOREIGN_UPPER = "QWX"
_FOREIGN_LOWER = "qwx"

_TR_VOWELS_LOWER = "aeıioöuü"
_TR_VOWELS_UPPER = "AEIİOÖUÜ"

# Vowel harmony classes, needed by later grammars (ordinals, suffixation).
_TR_FRONT_VOWELS = "eiöü"
_TR_BACK_VOWELS = "aıou"
_TR_ROUND_VOWELS = "oöuü"
_TR_UNROUND_VOWELS = "aeıi"

TO_LOWER = pynini.union(
    *[pynini.cross(u, l) for u, l in zip(_TR_ALPHA_UPPER + _FOREIGN_UPPER, _TR_ALPHA_LOWER + _FOREIGN_LOWER)]
).optimize()
TO_UPPER = pynini.invert(TO_LOWER).optimize()

TR_LOWER = pynini.union(*(_TR_ALPHA_LOWER + _FOREIGN_LOWER)).optimize()
TR_UPPER = pynini.union(*(_TR_ALPHA_UPPER + _FOREIGN_UPPER)).optimize()
TR_ALPHA = pynini.union(TR_LOWER, TR_UPPER).optimize()
TR_ALNUM = pynini.union(byte.DIGIT, TR_ALPHA).optimize()

TR_VOWELS = pynini.union(*(_TR_VOWELS_LOWER + _TR_VOWELS_UPPER)).optimize()
TR_CONSONANTS = pynini.difference(TR_ALPHA, TR_VOWELS).optimize()
TR_FRONT_VOWELS = pynini.union(*_TR_FRONT_VOWELS).optimize()
TR_BACK_VOWELS = pynini.union(*_TR_BACK_VOWELS).optimize()
TR_ROUND_VOWELS = pynini.union(*_TR_ROUND_VOWELS).optimize()
TR_UNROUND_VOWELS = pynini.union(*_TR_UNROUND_VOWELS).optimize()

# Turkish writes the decimal separator as a comma and groups thousands with a full stop.
TR_DECIMAL_SEPARATOR = ","
TR_THOUSANDS_SEPARATOR = "."
TR_COMMA_WORD = "virgül"
TR_MINUS_WORD = "eksi"

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")


def tr_lower(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Lower cases an fst using Turkish casing rules (``I`` -> ``ı``, ``İ`` -> ``i``).

    Args:
        fst: input fst

    Returns output fst with all Turkish upper case letters lower cased
    """
    return fst @ pynini.cdrewrite(TO_LOWER, "", "", NEMO_SIGMA)


def tr_upper(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Upper cases an fst using Turkish casing rules (``i`` -> ``İ``, ``ı`` -> ``I``).

    Args:
        fst: input fst

    Returns output fst with all Turkish lower case letters upper cased
    """
    return fst @ pynini.cdrewrite(TO_UPPER, "", "", NEMO_SIGMA)
