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

# The four criteria the owner named, as the starting revision. Editing these at
# runtime is an MCP call, not a deploy: see HousingStore.save_requirements.
DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "bedrooms": {"operator": "at_least", "value": 2},
    "bathrooms": {"operator": "at_least", "value": 2},
    "tv": {"minimum_class": "large"},
    "monthly_rent_thb": {"min": 20000, "max": 40000},
}

# Ordered smallest to largest so a minimum can be expressed as a position.
TV_CLASSES: tuple[str, ...] = ("none", "small", "medium", "large")


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
    """The verdict for a listing and the reasoning that produced it."""

    verdict: Verdict
    fields: tuple[FieldVerdict, ...]

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        """Criteria nobody could answer yet."""
        return tuple(f.field for f in self.fields if f.state is FieldState.UNKNOWN)

    def as_dict(self) -> dict[str, Any]:
        """Serializable form stored in housing_matches."""
        return {
            "verdict": self.verdict.value,
            "fields": [field.as_dict() for field in self.fields],
        }


class RequirementsError(ValueError):
    """A proposed requirements document is not usable."""


def validate_requirements(definition: dict[str, Any]) -> dict[str, Any]:
    """Check a requirements document before it can become active.

    Validation is strict and total: an unknown key is refused rather than
    ignored, because a silently dropped criterion is a filter that quietly
    stops filtering on the thing the owner just asked for.
    """
    if not isinstance(definition, dict):
        raise RequirementsError("requirements must be an object")

    allowed = {"bedrooms", "bathrooms", "tv", "monthly_rent_thb"}
    unknown = set(definition) - allowed
    if unknown:
        raise RequirementsError(
            f"unknown requirement(s): {', '.join(sorted(unknown))}; "
            f"supported: {', '.join(sorted(allowed))}"
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

    if "tv" in definition:
        spec = definition["tv"]
        if not isinstance(spec, dict):
            raise RequirementsError(f"tv must be {{'minimum_class': one of {TV_CLASSES}}}")
        minimum = spec.get("minimum_class")
        if minimum not in TV_CLASSES:
            raise RequirementsError(f"tv minimum_class must be one of {TV_CLASSES}")
        cleaned["tv"] = {"minimum_class": minimum}

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

    if not cleaned:
        raise RequirementsError("requirements are empty; at least one criterion is needed")
    return cleaned


def match_requirements(facts: dict[str, Any], requirements: dict[str, Any]) -> MatchResult:
    """Judge one listing's facts against one revision of the requirements.

    A pure function over already-extracted facts: no model call, no network.
    That is what makes re-judging the whole archive after an edit free, and
    why the re-match is deliberately not windowed.
    """
    fields: list[FieldVerdict] = []

    for field in ("bedrooms", "bathrooms"):
        spec = requirements.get(field)
        if spec is None:
            continue
        fields.append(_at_least(field, facts.get(field), int(spec["value"]), facts))

    tv_spec = requirements.get("tv")
    if tv_spec is not None:
        fields.append(_tv(facts, str(tv_spec["minimum_class"])))

    rent_spec = requirements.get("monthly_rent_thb")
    if rent_spec is not None:
        fields.append(_rent(facts.get("monthly_price_thb"), rent_spec))

    states = {field.state for field in fields}
    if FieldState.VIOLATED in states:
        verdict = Verdict.HARD_MISS
    elif FieldState.UNKNOWN in states:
        verdict = Verdict.POSSIBLE
    else:
        verdict = Verdict.CONFIRMED
    return MatchResult(verdict=verdict, fields=tuple(fields))


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
