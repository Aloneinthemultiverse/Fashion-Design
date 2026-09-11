"""Body reference by celebrity name.

The problem statement's primary input is not a photo: it is a *Western celebrity body
profile provided by the user* -- "dress me the way an Indian celebrity with Zendaya's
proportions dresses". This resolves such a name to body geometry.

Treating a named reference as `user_confirmed` is deliberate. Photo inference is a guess
the system makes and the user may correct; a named reference is a statement the user
made, so it is evidence of the same standing as a correction and must not be second-
guessed or blended with a photo reading.

Matching is forgiving because users type names the way they remember them -- wrong case,
missing diacritics, surname only. A near-miss that silently returns nothing is worse than
one that asks; `suggest()` exists so the caller can offer alternatives instead of an
empty result.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from fashion.core.models import BodyMetrics, CelebrityProfile


def normalise(name: str) -> str:
    """Casefold and strip accents so 'Aishwarya Rāi' matches 'aishwarya rai'."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


@dataclass(frozen=True, slots=True)
class ReferenceMatch:
    profile: CelebrityProfile
    exact: bool

    def to_metrics(self) -> BodyMetrics:
        """Body geometry implied by this celebrity.

        confidence is 1.0 and user_confirmed is True: the user named this person, so
        their proportions are a given for the request rather than something inferred.
        """
        return BodyMetrics(
            shape=self.profile.shape,
            build=self.profile.build,
            height_band=self.profile.height_band,
            confidence=1.0,
            user_confirmed=True,
        )


class BodyReferenceResolver:
    """Resolves a celebrity name to their recorded body profile."""

    def __init__(self, profiles: dict[str, CelebrityProfile]) -> None:
        self._by_name: dict[str, CelebrityProfile] = {}
        for profile in profiles.values():
            # First writer wins, so a later duplicate name cannot displace an earlier
            # profile and silently change what a stable query returns.
            self._by_name.setdefault(normalise(profile.name), profile)

    def resolve(self, name: str) -> ReferenceMatch | None:
        """Find the celebrity, exactly or by a confident partial match."""
        query = normalise(name)
        if not query:
            return None

        exact = self._by_name.get(query)
        if exact is not None:
            return ReferenceMatch(exact, exact=True)

        # Substring match, but only when it is unambiguous. Returning an arbitrary one
        # of several candidates would hand the user someone else's proportions without
        # telling them.
        candidates = [p for key, p in self._by_name.items() if query in key or key in query]
        if len(candidates) == 1:
            return ReferenceMatch(candidates[0], exact=False)
        return None

    def suggest(self, name: str, *, limit: int = 5) -> tuple[str, ...]:
        """Names worth offering when `resolve` fails.

        Matches on any word of the query, so a surname alone still surfaces candidates.
        """
        words = set(normalise(name).split())
        if not words:
            return ()
        hits = [profile.name for key, profile in self._by_name.items() if words & set(key.split())]
        return tuple(sorted(hits)[:limit])

    def __len__(self) -> int:
        return len(self._by_name)
