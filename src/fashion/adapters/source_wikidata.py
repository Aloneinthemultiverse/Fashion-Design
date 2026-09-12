"""Wikidata celebrity directory.

Supplies the *roster* -- who exists, where they are from, and a Commons image for each --
which per-name Commons category lookups cannot do at scale. One SPARQL query enumerates
thousands of actors with images already attached, so building a 1,000-2,000 person
directory is a handful of requests rather than thousands.

What Wikidata cannot supply is body shape, build or height band. Those are visual
judgements and are filled in later by the VLM labelling pass; profiles arrive here
unlabelled by design rather than guessed at.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

log = logging.getLogger(__name__)

SPARQL = "https://query.wikidata.org/sparql"

USER_AGENT = (
    "FashionRecommender/0.1 "
    "(https://github.com/example/fashion-design; research prototype) urllib/python"
)

# Wikidata entity ids used below, named so the queries stay readable.
HUMAN = "wd:Q5"
ACTOR = "wd:Q33999"
MODEL = "wd:Q4610556"
FEMALE = "wd:Q6581072"
MALE = "wd:Q6581097"

REGIONS: dict[str, str] = {
    "indian": "wd:Q668",
    "american": "wd:Q30",
    "british": "wd:Q145",
}


@dataclass(frozen=True, slots=True)
class CelebrityCandidate:
    """A person from the directory, before any visual labelling has happened."""

    qid: str
    name: str
    region: str
    image_url: str
    gender: str = ""
    height_cm: float | None = None
    # Wikidata P2003. The scraper addresses accounts by handle, not display name, so
    # without this the dynamic Instagram layer has nothing to work with.
    instagram: str = ""

    @property
    def id(self) -> str:
        return self.qid.rsplit("/", 1)[-1]


class WikidataCelebrityDirectory:
    """Enumerates celebrities with Commons-hosted images."""

    def __init__(self, *, delay_seconds: float = 1.0, timeout: float = 120.0) -> None:
        # Wikidata's public endpoint is shared infrastructure and throttles aggressively;
        # a second between queries keeps us well inside its limits.
        self._delay = delay_seconds
        self._timeout = timeout

    def _query(self, sparql: str, *, attempts: int = 4) -> list[dict[str, dict[str, str]]]:
        """Run a SPARQL query, retrying on the endpoint's transient failures.

        The public endpoint returns 502 and 429 under load rather than queueing, and a
        long roster run will hit both. Backoff is exponential because retrying a
        throttled endpoint immediately is what earns a longer block.
        """
        url = f"{SPARQL}?{urllib.parse.urlencode({'query': sparql, 'format': 'json'})}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        )
        for attempt in range(attempts):
            time.sleep(self._delay)
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.load(response)
                bindings: list[dict[str, dict[str, str]]] = payload["results"]["bindings"]
                return bindings
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == attempts - 1:
                    log.error("SPARQL query failed after %d attempts: %s", attempts, exc)
                    raise
                backoff = 2.0 * (2**attempt)
                log.warning(
                    "SPARQL attempt %d failed (%s); retrying in %.0fs", attempt + 1, exc, backoff
                )
                time.sleep(backoff)
        return []

    def fetch_region(
        self,
        region: str,
        *,
        limit: int = 500,
        offset: int = 0,
        gender: str | None = None,
        occupation: str = ACTOR,
    ) -> Iterator[CelebrityCandidate]:
        """Yield celebrities from one country.

        `offset` exists so large rosters can be paged; the endpoint times out on very
        large single result sets, and paging is cheaper than retrying a failed query.
        """
        if region not in REGIONS:
            raise ValueError(f"unknown region {region!r}; known: {sorted(REGIONS)}")

        gender_clause = f"; wdt:P21 {gender} " if gender else ""
        # One occupation per query, not `VALUES ?occ { actor model }`. The VALUES form
        # forces the planner to union two large sets and measured ~5x slower, which is
        # enough to push the public endpoint into 502s once paging is involved.
        #
        # ORDER BY makes paging stable -- without it the endpoint may return overlapping
        # or missing rows across offsets.
        sparql = f"""
        SELECT ?person ?personLabel ?image ?height ?instagram WHERE {{
          ?person wdt:P31 {HUMAN} ;
                  wdt:P106 {occupation} ;
                  wdt:P27 {REGIONS[region]} {gender_clause};
                  wdt:P18 ?image .
          OPTIONAL {{ ?person wdt:P2048 ?height . }}
          OPTIONAL {{ ?person wdt:P2003 ?instagram . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?person
        LIMIT {limit} OFFSET {offset}
        """

        for row in self._query(sparql):
            qid = row["person"]["value"]
            name = row.get("personLabel", {}).get("value", "")
            # An unresolved label falls back to the bare Q-id, which is useless as a
            # celebrity name and signals the entity has no English label.
            if not name or (name.startswith("Q") and name[1:].isdigit()):
                continue
            height_raw = row.get("height", {}).get("value")
            yield CelebrityCandidate(
                qid=qid,
                name=name,
                region=region,
                image_url=row["image"]["value"],
                gender=gender or "",
                height_cm=float(height_raw) if height_raw else None,
                instagram=row.get("instagram", {}).get("value", "").strip().lstrip("@"),
            )

    def fetch_roster(
        self,
        targets: dict[str, int],
        *,
        page_size: int = 500,
        gender: str | None = None,
        occupations: tuple[str, ...] = (ACTOR, MODEL),
    ) -> list[CelebrityCandidate]:
        """Collect a roster across regions, de-duplicated by Q-id.

        Dual citizenship means the same person can appear under two regions; the first
        region wins so counts stay honest.
        """
        seen: set[str] = set()
        roster: list[CelebrityCandidate] = []

        for region, wanted in targets.items():
            collected = 0
            # Actors first, then models: actors are the far larger and better-labelled
            # set, so the roster stays dominated by them and models only top it up.
            for occupation in occupations:
                offset = 0
                while collected < wanted:
                    page = list(
                        self.fetch_region(
                            region,
                            limit=min(page_size, wanted - collected + 50),
                            offset=offset,
                            gender=gender,
                            occupation=occupation,
                        )
                    )
                    if not page:
                        break
                    for candidate in page:
                        if candidate.id in seen or collected >= wanted:
                            continue
                        seen.add(candidate.id)
                        roster.append(candidate)
                        collected += 1
                    offset += len(page)
                if collected >= wanted:
                    break
            if collected < wanted:
                log.warning("region %r yielded %d of %d requested", region, collected, wanted)

        return roster
