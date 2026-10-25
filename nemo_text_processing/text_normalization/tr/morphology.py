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
The shared Turkish inflection layer of the text normalization grammars.

It attaches suffixes to spoken readings: the ordinal suffix of OrdinalFst, the
locative of FractionFst's denominator, and the suffixes written after an apostrophe
that SuffixFst reads on numbers, dates, times, percentages, units, currency words,
acronyms and addresses.

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

On top of these, the module provides:

    - a deliberately limited inventory of productive suffixes (``SuffixSpec``): the
      locative, ablative, dative, accusative, genitive, instrumental, plural, third
      person possessive and ordinal, with their buffer consonants (y, n, s) spelled
      out in each suffix's templates, and the derivational -lI and -lIk;
    - lexical stem alternation, applied only where a table lists it: the numerals'
      "dört" -> "dörd" before a vowel initial suffix, and no general softening;
    - harmony exceptions, words whose suffixes do not follow their last vowel;
    - suffix validation against a spoken stem (``suffix_validator``), and against a
      separate spelling anchor where a written form is spelled after one reading and
      spoken as another (``inflect_by_anchor``, "TL'ye" -> "yüz liraya").

This is productive morphology for text normalization, not general purpose Turkish
morphology: there is no analysis of arbitrary words, no derivational morphology
beyond -lI and -lIk and no compounding rules. Only the suffix attached to a stem is modelled; the suffixes
stacked after it are spelled as they are spoken, and SuffixFst reads them as written.
"""

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    _TR_VOICELESS_CONSONANTS,
    _TR_VOWELS_LOWER,
    NEMO_SIGMA,
    NEMO_SPACE,
    TR_ALPHA,
    TR_CONSONANTS,
    TR_LOWER,
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

# Words whose suffixes do not harmonize with their last vowel, with the vowel their
# suffixes harmonize with instead. These are lexical properties of the words, so they
# apply wherever a spoken stem ends in one of them. TDK records "saat" ("saat, -ti"),
# "jul" ("jul, -lü") and "general", "orgeneral", "korgeneral", "tümgeneral",
# "tuğgeneral" ("-li"); "kilovatsaat" is listed because it ends in "saat" and its
# suffixes follow that word.
HARMONY_EXCEPTIONS = load_labels(get_abs_path("data/morphology/harmony_exceptions.tsv"))


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


class SuffixSpec:
    """
    One Turkish suffix, described by its templates per final segment class of the
    stem, as for ``assimilating_suffix``: ``{vowel}`` marks the harmonic vowel.

    Written after an apostrophe, a suffix is spelled exactly as it is spoken, so one
    description gives both the written allomorph that is correct for a stem and the
    spoken inflected form. Only the stem may differ between the two: a numeral
    alternates before a vowel initial suffix ("4'e" -> "dörde"), see ``inflect``.

    Args:
        name: short name, e.g. "locative"
        harmony: harmony table, LOW_VOWEL_HARMONY or HIGH_VOWEL_HARMONY
        after_vowel: template after a vowel final stem
        after_voiced: template after a voiced consonant
        after_voiceless: template after a voiceless consonant
    """

    def __init__(self, name: str, harmony, after_vowel: str, after_voiced: str, after_voiceless: str):
        self.name = name
        self.harmony = harmony
        self.after_vowel = after_vowel
        self.after_voiced = after_voiced
        self.after_voiceless = after_voiceless

    def attach(self, prefix: str = "", exceptions=None) -> "pynini.FstLike":
        """
        Appends the suffix, preceded by ``prefix``, to a stem: "altı" -> "altıda".

        Args:
            prefix: written before the suffix, e.g. an apostrophe
            exceptions: (word, vowel) pairs, e.g. HARMONY_EXCEPTIONS: a stem whose last
                word is listed harmonizes with the given vowel, "saat" -> "saatte"

        Returns a pynini.FstLike
        """
        templates = [prefix + self.after_vowel, prefix + self.after_voiced, prefix + self.after_voiceless]
        regular = _suffix_by_final_segment(self.harmony, *templates)
        if not exceptions:
            return regular

        def ending_in(word: str) -> "pynini.FstLike":
            return pynini.closure(pynini.closure(_STEM_CHAR) + NEMO_SPACE, 0, 1) + word

        harmony = dict(self.harmony)
        listed = pynini.union(*[ending_in(word) for word, _ in exceptions])
        branches = [pynini.compose(pynini.difference(pynini.closure(_STEM_CHAR), listed), regular)]
        for word, vowel in exceptions:
            final = word[-1]
            template = templates[0 if final in _TR_VOWELS_LOWER else 2 if final in _TR_VOICELESS_CONSONANTS else 1]
            branches.append(ending_in(word) + pynutil.insert(template.format(vowel=harmony[vowel])))
        return pynini.union(*branches).optimize()

    def variants(self, template: str) -> set:
        """
        The spellings a writer may use for one template, whatever the stem's vowel:
        every vowel of the harmony table, and for a suffix beginning with "d" or "t" the
        other consonant too. "60'dan" is written for "60'tan" and "11.00'dan" for
        "11.00'den". The buffer consonants y, n and s are never exchanged.

        Args:
            template: one of the three templates

        Returns a set of strings
        """
        forms = {template.format(vowel=vowel) for vowel in sorted({vowel for _, vowel in self.harmony})}
        forms |= {{"d": "t", "t": "d"}[form[0]] + form[1:] for form in forms if form[:1] in ("d", "t")}
        return forms

    def attach_lenient(self, prefix: str = "") -> "pynini.FstLike":
        """
        Appends to a stem, preceded by ``prefix``, any of the spellings ``variants``
        accepts for the stem's final segment class: "altmış" -> "altmış'tan",
        "altmış'dan", "altmış'ten" or "altmış'den". The correct allomorph is one of them.

        Args:
            prefix: written before the suffix, e.g. an apostrophe

        Returns a pynini.FstLike
        """
        anything = pynini.closure(_STEM_CHAR)
        vowels = pynini.union(*_TR_VOWELS_LOWER)

        def spellings(template: str) -> "pynini.FstLike":
            return pynutil.insert(pynini.union(*[prefix + form for form in sorted(self.variants(template))]))

        return pynini.union(
            anything + vowels + spellings(self.after_vowel),
            anything + TR_VOICED_CONSONANTS + spellings(self.after_voiced),
            anything + TR_VOICELESS_CONSONANTS + spellings(self.after_voiceless),
        ).optimize()

    @property
    def vowel_initial_after_consonant(self) -> bool:
        """Whether the suffix begins with a vowel after a consonant final stem, which is
        where a stem such as "dört" alternates."""
        return self.after_voiceless.startswith("{vowel}")


LOCATIVE = SuffixSpec("locative", LOW_VOWEL_HARMONY, "d{vowel}", "d{vowel}", "t{vowel}")
ABLATIVE = SuffixSpec("ablative", LOW_VOWEL_HARMONY, "d{vowel}n", "d{vowel}n", "t{vowel}n")
DATIVE = SuffixSpec("dative", LOW_VOWEL_HARMONY, "y{vowel}", "{vowel}", "{vowel}")
ACCUSATIVE = SuffixSpec("accusative", HIGH_VOWEL_HARMONY, "y{vowel}", "{vowel}", "{vowel}")
GENITIVE = SuffixSpec("genitive", HIGH_VOWEL_HARMONY, "n{vowel}n", "{vowel}n", "{vowel}n")
INSTRUMENTAL = SuffixSpec("instrumental", LOW_VOWEL_HARMONY, "yl{vowel}", "l{vowel}", "l{vowel}")
PLURAL = SuffixSpec("plural", LOW_VOWEL_HARMONY, "l{vowel}r", "l{vowel}r", "l{vowel}r")
POSSESSIVE_3SG = SuffixSpec("possessive_3sg", HIGH_VOWEL_HARMONY, "s{vowel}", "{vowel}", "{vowel}")
ORDINAL = SuffixSpec("ordinal", HIGH_VOWEL_HARMONY, "nc{vowel}", "{vowel}nc{vowel}", "{vowel}nc{vowel}")

# The case, number and possessive suffixes, as stem -> inflected stem:
#   ablative -DAn        altı -> altıdan, beş -> beşten
#   dative -(y)A         altı -> altıya, on -> ona
#   accusative -(y)I     iki -> ikiyi, on -> onu
#   genitive -(n)In      iki -> ikinin, beş -> beşin
#   instrumental -(y)lA  iki -> ikiyle, beş -> beşle
#   plural -lAr          iki -> ikiler, on -> onlar
#   possessive -(s)I     iki -> ikisi, üç -> üçü
ABLATIVE_SUFFIX = ABLATIVE.attach()
DATIVE_SUFFIX = DATIVE.attach()
ACCUSATIVE_SUFFIX = ACCUSATIVE.attach()
GENITIVE_SUFFIX = GENITIVE.attach()
INSTRUMENTAL_SUFFIX = INSTRUMENTAL.attach()
PLURAL_SUFFIX = PLURAL.attach()
POSSESSIVE_3SG_SUFFIX = POSSESSIVE_3SG.attach()

CASE_SUFFIXES = [LOCATIVE, ABLATIVE, DATIVE, ACCUSATIVE, GENITIVE, INSTRUMENTAL, PLURAL, POSSESSIVE_3SG]

# The two derivational suffixes written after a number, unit or abbreviation, both
# beginning with a consonant and so never alternating the stem:
#   -lI     iki -> ikili, on -> onlu, dört -> dörtlü ("2000'li yıllar")
#   -lIk    beş -> beşlik, yüz -> yüzlük, dört -> dörtlük ("5 kg'lık")
WITH = SuffixSpec("with", HIGH_VOWEL_HARMONY, "l{vowel}", "l{vowel}", "l{vowel}")
MEASURE_OF = SuffixSpec("measure_of", HIGH_VOWEL_HARMONY, "l{vowel}k", "l{vowel}k", "l{vowel}k")
DERIVATIONAL_SUFFIXES = [WITH, MEASURE_OF]

# The one lexical stem alternation of the numerals, dört -> dörd, before a suffix that
# begins with a vowel ("dörde", "dördü", "dördüncü"). Turkish consonant softening is
# lexical, so nothing else alternates: "kırka", "üçü", and no acronym or unit softens
# ("tübitakın").
NUMERAL_STEM_ALTERNATION = ORDINAL_STEM_ALTERNATION


def inflect(spec: SuffixSpec, stem_alternation: "pynini.FstLike" = None, exceptions=None) -> "pynini.FstLike":
    """
    Inflects a spoken stem, applying ``stem_alternation`` first where the suffix
    begins with a vowel after a consonant:

        inflect(DATIVE, NUMERAL_STEM_ALTERNATION): "dört" -> "dörde", "kırk" -> "kırka"
        inflect(LOCATIVE, NUMERAL_STEM_ALTERNATION): "dört" -> "dörtte"

    Args:
        spec: the suffix
        stem_alternation: lexical alternation of the stem's final word, or None
        exceptions: harmony exceptions, see SuffixSpec.attach

    Returns a pynini.FstLike
    """
    attached = spec.attach(exceptions=exceptions)
    if stem_alternation is not None and spec.vowel_initial_after_consonant:
        return pynini.compose(stem_alternation, attached).optimize()
    return attached


def written_suffix(spec: SuffixSpec, apostrophe: str = "'", exceptions=None) -> "pynini.FstLike":
    """
    Appends the written form of a suffix to a spoken anchor, apostrophe included:
    "te le" -> "te le'ye". The anchor is the pronunciation Turkish spelling chooses the
    suffix by; it is not necessarily what is spoken (see ``inflect_by_anchor``).
    """
    return spec.attach(prefix=apostrophe, exceptions=exceptions)


def suffix_validator(
    specs, stem_alternation: "pynini.FstLike" = None, apostrophe: str = "'", exceptions=None, lenient: bool = False
) -> "pynini.FstLike":
    """
    Maps a spoken stem followed by an apostrophe and a written suffix to the inflected
    stem, accepting only the written allomorph the stem selects:

        "iki bin yirmi altı'da" -> "iki bin yirmi altıda"
        "iki bin yirmi altı'de" -> rejected
        "dört'e" -> "dörde" (with NUMERAL_STEM_ALTERNATION)

    With ``lenient``, a suffix written with another vowel or with "d" and "t" exchanged
    is also accepted and read correctly, see ``SuffixSpec.variants``:

        "altmış'dan" -> "altmıştan"

    Where the anchor is the spoken stem, this is the whole of the suffix logic: the
    written suffix is checked against the stem and the output inflects the same stem.

    Args:
        specs: the suffixes to accept
        stem_alternation: lexical alternation of the stem's final word, or None
        apostrophe: the apostrophe expected in the input
        exceptions: harmony exceptions, see SuffixSpec.attach
        lenient: also accept the spellings of ``SuffixSpec.variants``

    Returns a pynini.FstLike
    """

    def written(spec: SuffixSpec) -> "pynini.FstLike":
        if lenient:
            return spec.attach_lenient(prefix=apostrophe)
        return written_suffix(spec, apostrophe, exceptions)

    return pynini.union(
        *[pynini.compose(pynini.invert(written(spec)), inflect(spec, stem_alternation, exceptions)) for spec in specs]
    ).optimize()


def inflect_by_anchor(
    anchor: str,
    specs,
    stem_alternation: "pynini.FstLike" = None,
    apostrophe: str = "'",
    exceptions=None,
    lenient: bool = False,
):
    """
    For a written form whose suffix is chosen by a different pronunciation than the one
    spoken, e.g. a currency code: "TL'ye" is spelled after "te le", but "100 TL'ye" is
    read "yüz liraya". Returns a transducer that deletes an apostrophe and the written
    suffix ``anchor`` selects, and appends the suffix to the spoken stem before it:

        inflect_by_anchor("te le", [DATIVE]) applied to "yüz lira'ye" -> "yüz liraya"

    Args:
        anchor: the pronunciation the written suffix agrees with
        specs: the suffixes to accept
        stem_alternation: lexical alternation of the spoken stem, or None
        apostrophe: the apostrophe expected in the input
        lenient: also accept the spellings of ``SuffixSpec.variants`` for the anchor

    Returns a pynini.FstLike from a spoken stem followed by the written suffix
    """
    from pynini.lib import rewrite

    branches = []
    for spec in specs:
        written = spec.attach_lenient(prefix=apostrophe) if lenient else written_suffix(spec, apostrophe, exceptions)
        suffixes = [rewritten[len(anchor) :] for rewritten in rewrite.rewrites(anchor, written)]
        branches.append(
            pynini.compose(
                pynini.closure(_STEM_CHAR, 1) + pynutil.delete(pynini.union(*suffixes)),
                inflect(spec, stem_alternation, exceptions),
            )
        )
    return pynini.union(*branches).optimize()


def _harmonize(state: str, vowel: str) -> str:
    """The vowel a suffix spells for ``vowel`` after a syllable with vowel ``state``."""
    back, rounded = state in "aıou", state in "oöuü"
    if vowel in "ae":
        return "a" if back else "e"
    if vowel in "ıiuü":
        return {(True, False): "ı", (True, True): "u", (False, False): "i", (False, True): "ü"}[(back, rounded)]
    return vowel


def stacked_suffixes(max_vowels: int = 4) -> dict:
    """
    For each vowel a suffixed reading may end with, a transducer that spells the
    suffixes stacked after it in vowel harmony with it, left to right: after
    "altılar", the written "i" of "Il-76'leri" is "ı" -> "altıları". Stacked suffixes
    spelled in harmony are kept as written. The suffixes that do not harmonize keep
    their vowel: an "i" or "e" right after "k" (-ki, -ken, "yedideki") and "o" (-yor).

    Args:
        max_vowels: the most vowels the stacked suffixes may contain

    Returns a dict from a vowel to a pynini.FstLike over lower case letters
    """
    vowels = "aeıioöuü"
    consonant = pynini.difference(TR_LOWER, pynini.union(*vowels))
    consonants = pynini.closure(consonant)
    after_k = consonants + "k"
    not_after_k = pynini.difference(consonants, after_k)
    graphs = {}

    def graph(state: str, remaining: int) -> "pynini.FstLike":
        if (state, remaining) in graphs:
            return graphs[(state, remaining)]
        branches = [consonants]
        if remaining:
            for vowel in vowels:
                spoken = _harmonize(state, vowel)
                kept = vowel in "ie"
                branches.append(not_after_k + pynini.cross(vowel, spoken) + graph(spoken, remaining - 1))
                if kept:
                    branches.append(after_k + pynini.accep(vowel) + graph(vowel, remaining - 1))
                else:
                    branches.append(after_k + pynini.cross(vowel, spoken) + graph(spoken, remaining - 1))
        graphs[(state, remaining)] = pynini.union(*branches).optimize()
        return graphs[(state, remaining)]

    return {vowel: graph(vowel, max_vowels) for vowel in vowels}
