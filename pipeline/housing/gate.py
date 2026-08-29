"""The cheap gate that keeps the island's chatter out of the model.

Only general chats pass through here. A dedicated rentals board is read in
full: every message on it is a candidate, and the few that are not cost a
fraction of a cent to find out about.

The gate exists because the general chats are large. The island's main chat
has thirty thousand members; running an extraction on every message in it
would pay for a model call per conversation turn about scooter repairs. It is
deliberately wide — one housing word OR one price-shaped number is enough —
because the failure that matters is a listing dropped in silence, not a joke
about landlords sent to the model by mistake.

Measured 2026-08-29 against 1,200 consecutive messages from a chat that is
nothing but rental listings (`rent_phangan`): the gate drops 116, of which 15
are distinct, and none of them is an offer of a place to live. What it drops
there is hotel advertising, an investment spam run, and people ASKING for
housing, which is not an offer either. On the general chat it passes 19.5% of
messages, so four in five never reach the model.
"""

from __future__ import annotations

import re

# Any of these words is enough. The list is a net, not a definition: an
# advertisement is identified by the model, not here.
#
# Every stem is anchored at a word boundary, which Python's `re` computes over
# Unicode word characters, so it works on Russian. Without the anchors the
# stem "дом" matches inside "рядом" and the gate passes half the conversation
# on the island — measured, not theorised: it let through "Кто подскажет
# стоматолога рядом с Тонг Салой?".
HOUSING_HINTS = re.compile(
    r"\bсда(ю|м|ет|ёт)|\bсдаётся|\bсдается|\bаренд|\bснять|\bсниму|\bпересда"
    r"|\brent|\bavailable|\bдом|\bвилл|\bvilla|\bhouse|\bбунгало|\bbungalow"
    r"|\bапарт|\bapartment|\bквартир|\bстуди|\bstudio|\broom|\bкомнат|\bспальн"
    r"|\bbedroom|\bbr\b|\bтаунхаус|\bжиль",
    re.IGNORECASE,
)

# A price shaped like a rent, in any of the forms this corpus writes them.
# Four digits catches "15000" and "25 000" once spaces are stripped; the
# suffixed forms catch "15k", "30 тыс", "20к".
PRICE_HINTS = re.compile(r"\d{4,}|\d+\s*(k|к|тыс)|бат|baht|thb|฿", re.IGNORECASE)

# Short enough to admit a terse real listing — "Сдаю дом 2 спальни 25000 бат"
# is twenty-eight characters — while still dropping the one-word replies that
# make up most of a chat's traffic.
MIN_LENGTH = 20


def could_be_housing(text: str | None) -> bool:
    """Whether a general-chat message is worth reading properly.

    Deliberately generous. Every false positive costs one small model call
    that answers "not a rental offer"; every false negative is an
    advertisement the owner never hears about.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_LENGTH:
        return False
    compact = re.sub(r"(?<=\d)[\s ](?=\d)", "", stripped)
    return bool(HOUSING_HINTS.search(compact)) or bool(PRICE_HINTS.search(compact))
