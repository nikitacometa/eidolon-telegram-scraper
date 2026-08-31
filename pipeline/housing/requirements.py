"""What the owner is looking for, and how a listing is judged against it.

The judgement has three outcomes per field, not two. A Phangan advertisement
states its bathroom count 2.3% of the time and the size of its television
never, so a matcher that reads "not stated" as "does not match" would reject
almost every real listing and report nothing at all — the exact silence this
project has already been burned by once.

So: a field is satisfied, violated, or unknown, and only a violation can
reject a listing. Unknown fields are carried into the alert by name, so the
owner sees "2 bedrooms confirmed, bathrooms unknown, TV unknown" and decides
for himself whether to ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from storage.housing import Verdict

# The owner's criteria, as the starting revision. Editing these at runtime is
# an MCP call, not a deploy: see HousingStore.save_requirements.
#
# Two tiers. The top-level fields are HARD: a confirmed violation rejects the
# listing. "preferences" are SOFT: they can never reject anything — measured
# on this corpus a television size is never stated and a terrace only 43% of
# the time, so a hard requirement there is a filter that mostly filters on
# silence. Preferences contribute to a weighted score instead, which ranks
# listings and shows in the alert.
DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "bedrooms": {"operator": "at_least", "value": 2},
    "monthly_rent_thb": {"min": 20000, "max": 40000},
    "property_type": {"require": "house"},
    "preferences": {
        "tv": {"minimum_class": "large", "weight": 30},
        "terrace": {"weight": 25},
        "private_setting": {"weight": 25},
        "nature_setting": {"weight": 20},
    },
}

# Ordered smallest to largest so a minimum can be expressed as a position.
TV_CLASSES: tuple[str, ...] = ("none", "small", "medium", "large")

# What an advertisement can offer. "house" is a standalone building of its
# own (дом, вилла, бунгало, таунхаус); "apartment" is a unit inside a shared
# building (квартира, кондо, апартаменты, студия); "room" is a room in
# someone else's space; "hotel" is nightly-style accommodation.
PROPERTY_TYPES: tuple[str, ...] = ("house", "apartment", "room", "hotel")

# Preference keys and how each is judged.
PREFERENCE_FIELDS: tuple[str, ...] = ("tv", "terrace", "private_setting", "nature_setting")
DEFAULT_PREFERENCE_WEIGHT = 25


class FieldState(StrEnum):
    """How one criterion stands for one listing."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FieldVerdict:
    """One criterion's outcome, with the observation behind it."""

    field: str
    state: FieldState
    detail: str

    def as_dict(self) -> dict[str, str]:
        """Serializable form stored alongside the match."""
        return {"field": self.field, "state": self.state.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The verdict for a listing and the reasoning that produced it.

    ``fields`` are the hard criteria and alone decide the verdict.
    ``preferences`` never reject; they fold into ``preference_score`` —
    the weighted share of preferences the listing is CONFIRMED to satisfy,
    0..100. An unknown preference scores nothing and dooms nothing.
    """

    verdict: Verdict
    fields: tuple[FieldVerdict, ...]
    preferences: tuple[FieldVerdict, ...] = ()
    preference_score: int = 0

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        """Hard criteria nobody could answer yet."""
        return tuple(f.field for f in self.fields if f.state is FieldState.UNKNOWN)

    @property
    def unknown_preferences(self) -> tuple[str, ...]:
        """Preferences nobody could answer yet."""
        return tuple(f.field for f in self.preferences if f.state is FieldState.UNKNOWN)

    def as_dict(self) -> dict[str, Any]:
        """Serializable form stored in housing_matches."""
        return {
            "verdict": self.verdict.value,
            "fields": [field.as_dict() for field in self.fields],
            "preferences": [field.as_dict() for field in self.preferences],
            "preference_score": self.preference_score,
        }


class RequirementsError(ValueError):
    """A proposed requirements document is not usable."""


def normalize_requirements(definition: dict[str, Any]) -> dict[str, Any]:
    """Bring an older revision into the two-tier shape without editing it.

    Revisions are append-only history; the ones written before preferences
    existed carry "tv" at the top level. The owner's meaning for a television
    was always "want one", not "reject everything silent about one" — a size
    is stated in 0% of this corpus — so a top-level tv reads as a preference.
    """
    if not isinstance(definition, dict):
        return definition
    if "tv" not in definition:
        return definition
    migrated = {key: value for key, value in definition.items() if key != "tv"}
    preferences = dict(migrated.get("preferences") or {})
    tv_spec = dict(definition["tv"]) if isinstance(definition["tv"], dict) else {}
    tv_spec.setdefault("weight", 30)
    preferences.setdefault("tv", tv_spec)
    migrated["preferences"] = preferences
    return migrated


def validate_requirements(definition: dict[str, Any]) -> dict[str, Any]:
    """Check a requirements document before it can become active.

    Validation is strict and total: an unknown key is refused rather than
    ignored, because a silently dropped criterion is a filter that quietly
    stops filtering on the thing the owner just asked for.
    """
    if not isinstance(definition, dict):
        raise RequirementsError("requirements must be an object")
    definition = normalize_requirements(definition)

    allowed = {"bedrooms", "bathrooms", "monthly_rent_thb", "property_type", "preferences"}
    unknown = set(definition) - allowed
    if unknown:
        raise RequirementsError(
            f"unknown requirement(s): {', '.join(sorted(unknown))}; "
            f"supported: {', '.join(sorted(allowed))} "
            "(tv/terrace/private_setting/nature_setting live under 'preferences')"
        )

    cleaned: dict[str, Any] = {}
    for field in ("bedrooms", "bathrooms"):
        if field not in definition:
            continue
        spec = definition[field]
        if not isinstance(spec, dict) or spec.get("operator") != "at_least":
            raise RequirementsError(f"{field} must be {{'operator': 'at_least', 'value': N}}")
        value = spec.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RequirementsError(f"{field} value must be a non-negative integer")
        cleaned[field] = {"operator": "at_least", "value": value}

    if "property_type" in definition:
        spec = definition["property_type"]
        if not isinstance(spec, dict) or spec.get("require") not in PROPERTY_TYPES:
            raise RequirementsError(f"property_type must be {{'require': one of {PROPERTY_TYPES}}}")
        cleaned["property_type"] = {"require": spec["require"]}

    if "preferences" in definition:
        prefs = definition["preferences"]
        if not isinstance(prefs, dict):
            raise RequirementsError("preferences must be an object")
        unknown_prefs = set(prefs) - set(PREFERENCE_FIELDS)
        if unknown_prefs:
            raise RequirementsError(
                f"unknown preference(s): {', '.join(sorted(unknown_prefs))}; "
                f"supported: {', '.join(PREFERENCE_FIELDS)}"
            )
        cleaned_prefs: dict[str, Any] = {}
        for name, spec in prefs.items():
            if not isinstance(spec, dict):
                raise RequirementsError(f"preference {name} must be an object")
            weight = spec.get("weight", DEFAULT_PREFERENCE_WEIGHT)
            if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= 100:
                raise RequirementsError(f"preference {name} weight must be an integer 0..100")
            cleaned_spec: dict[str, Any] = {"weight": weight}
            if name == "tv":
                minimum = spec.get("minimum_class", "large")
                if minimum not in TV_CLASSES:
                    raise RequirementsError(f"tv minimum_class must be one of {TV_CLASSES}")
                cleaned_spec["minimum_class"] = minimum
            elif set(spec) - {"weight"}:
                extra = ", ".join(sorted(set(spec) - {"weight"}))
                raise RequirementsError(f"preference {name} accepts only 'weight', got: {extra}")
            cleaned_prefs[name] = cleaned_spec
        if cleaned_prefs:
            cleaned["preferences"] = cleaned_prefs

    if "monthly_rent_thb" in definition:
        spec = definition["monthly_rent_thb"]
        if not isinstance(spec, dict):
            raise RequirementsError("monthly_rent_thb must be {'min': N, 'max': M}")
        bounds: dict[str, int] = {}
        for bound in ("min", "max"):
            value = spec.get(bound)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RequirementsError(f"monthly_rent_thb {bound} must be a non-negative integer")
            bounds[bound] = value
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise RequirementsError("monthly_rent_thb min must not exceed max")
        cleaned["monthly_rent_thb"] = bounds

    if not cleaned or not (set(cleaned) - {"preferences"}):
        raise RequirementsError(
            "requirements need at least one hard criterion "
            "(bedrooms, bathrooms, monthly_rent_thb or property_type)"
        )
    return cleaned


def match_requirements(facts: dict[str, Any], requirements: dict[str, Any]) -> MatchResult:
    """Judge one listing's facts against one revision of the requirements.

    A pure function over already-extracted facts: no model call, no network.
    That is what makes re-judging the whole archive after an edit free, and
    why the re-match is deliberately not windowed.
    """
    requirements = normalize_requirements(requirements)
    fields: list[FieldVerdict] = []

    for field in ("bedrooms", "bathrooms"):
        spec = requirements.get(field)
        if spec is None:
            continue
        fields.append(_at_least(field, facts.get(field), int(spec["value"]), facts))

    type_spec = requirements.get("property_type")
    if type_spec is not None:
        fields.append(_property_type(facts, str(type_spec["require"])))

    rent_spec = requirements.get("monthly_rent_thb")
    if rent_spec is not None:
        fields.append(_rent(facts.get("monthly_price_thb"), rent_spec))

    preferences: list[FieldVerdict] = []
    weights: dict[str, int] = {}
    for name, spec in (requirements.get("preferences") or {}).items():
        weights[name] = int(spec.get("weight", DEFAULT_PREFERENCE_WEIGHT))
        if name == "tv":
            preferences.append(_tv(facts, str(spec.get("minimum_class", "large"))))
        else:
            preferences.append(_soft_presence(name, facts))

    total_weight = sum(weights.values())
    satisfied_weight = sum(
        weights[pref.field] for pref in preferences if pref.state is FieldState.SATISFIED
    )
    score = round(100 * satisfied_weight / total_weight) if total_weight else 0

    # Only the hard tier decides: a preference can neither reject a listing
    # nor keep one merely unknown out of CONFIRMED.
    states = {field.state for field in fields}
    if FieldState.VIOLATED in states:
        verdict = Verdict.HARD_MISS
    elif FieldState.UNKNOWN in states:
        verdict = Verdict.POSSIBLE
    else:
        verdict = Verdict.CONFIRMED
    return MatchResult(
        verdict=verdict,
        fields=tuple(fields),
        preferences=tuple(preferences),
        preference_score=score,
    )


def _property_type(facts: dict[str, Any], required: str) -> FieldVerdict:
    """Judge the kind of property, where only a stated wrong kind rejects.

    The extractor is guarded against fabricating a rejecting type, and a
    photograph can only ever CONFIRM a type (a frame of a building says
    nothing about which unit is offered), so a vision-sourced mismatch stays
    unknown rather than rejecting.
    """
    value = facts.get("property_type")
    if value is None:
        return FieldVerdict("property_type", FieldState.UNKNOWN, "not stated")
    if value == required:
        return FieldVerdict("property_type", FieldState.SATISFIED, str(value))
    source = str(facts.get("property_type_source") or "unknown")
    if source == "text":
        return FieldVerdict("property_type", FieldState.VIOLATED, f"{value} (want {required})")
    return FieldVerdict(
        "property_type", FieldState.UNKNOWN, f"{value} per photos only (want {required})"
    )


def _soft_presence(field: str, facts: dict[str, Any]) -> FieldVerdict:
    """Judge a nice-to-have that listings either claim or stay silent about.

    Silence is the overwhelming norm (a terrace is mentioned in 43% of this
    corpus, privacy in 29%, a nature setting in 23%), and nobody advertises
    the absence of a terrace — so the only meaningful states are "claimed"
    and "unknown". A stored False still maps to unknown: the extractor is
    instructed to produce true or null, never a guessed absence.
    """
    value = facts.get(field)
    if value in (1, True):
        return FieldVerdict(field, FieldState.SATISFIED, "claimed in the listing")
    return FieldVerdict(field, FieldState.UNKNOWN, "not mentioned")


def _at_least(field: str, value: Any, minimum: int, facts: dict[str, Any]) -> FieldVerdict:
    """Judge a count, treating a photograph's count as a lower bound.

    Vision can only ever say "I saw at least this many": a bathroom outside
    every frame is invisible, so a visual count below the requirement is not a
    violation — it is still unknown. Text is different; when someone writes
    "1 bedroom" they are describing the whole property.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return FieldVerdict(field, FieldState.UNKNOWN, "not stated")
    if value >= minimum:
        return FieldVerdict(field, FieldState.SATISFIED, f"{value} (need {minimum}+)")
    source = str(facts.get(f"{field}_source") or "unknown")
    if source == "vision":
        return FieldVerdict(
            field,
            FieldState.UNKNOWN,
            f"photos show {value}, which is a lower bound (need {minimum}+)",
        )
    return FieldVerdict(field, FieldState.VIOLATED, f"{value} (need {minimum}+)")


def _tv(facts: dict[str, Any], minimum_class: str) -> FieldVerdict:
    """Judge the television, where 'unclear' is a real and common answer."""
    size_class = facts.get("tv_size_class")
    present = facts.get("tv_present")

    if present is False or size_class == "none":
        # An explicit "no television" is a violation only when someone actually
        # said so; a photograph that failed to show one says nothing, and the
        # extractor is instructed to report that as unclear, not as none.
        source = str(facts.get("tv_source") or "unknown")
        if source in {"text", "vision"}:
            return FieldVerdict("tv", FieldState.VIOLATED, "no television reported")
        return FieldVerdict("tv", FieldState.UNKNOWN, "not stated")

    if size_class in (None, "unclear"):
        if present is True:
            return FieldVerdict("tv", FieldState.UNKNOWN, "television present, size unclear")
        return FieldVerdict("tv", FieldState.UNKNOWN, "not stated")

    if size_class not in TV_CLASSES:
        return FieldVerdict("tv", FieldState.UNKNOWN, f"unrecognized size class {size_class!r}")

    if TV_CLASSES.index(str(size_class)) >= TV_CLASSES.index(minimum_class):
        return FieldVerdict("tv", FieldState.SATISFIED, f"{size_class} (need {minimum_class}+)")
    # A television smaller than asked for is a real, stated fact about the
    # property, so this one does reject.
    return FieldVerdict("tv", FieldState.VIOLATED, f"{size_class} (need {minimum_class}+)")


def _rent(price: Any, bounds: dict[str, Any]) -> FieldVerdict:
    """Judge the asking price, which is stated in text or not at all."""
    if not isinstance(price, int) or isinstance(price, bool):
        return FieldVerdict("monthly_rent_thb", FieldState.UNKNOWN, "no price stated")
    low = bounds.get("min")
    high = bounds.get("max")
    if isinstance(low, int) and price < low:
        return FieldVerdict("monthly_rent_thb", FieldState.VIOLATED, f"{price} THB, below {low}")
    if isinstance(high, int) and price > high:
        return FieldVerdict("monthly_rent_thb", FieldState.VIOLATED, f"{price} THB, above {high}")
    window = f"{low or 0}–{high}" if high is not None else f"{low or 0}+"
    return FieldVerdict("monthly_rent_thb", FieldState.SATISFIED, f"{price} THB (want {window})")
