# Расширение taxonomy мест до services и medicine

## Что изменено

- `pipeline/indexer.py:26-104`: prompt version поднят с `places-v1` до `places-v2`.
  `PlaceType` теперь является typed `StrEnum`; к прежним типам добавлены `clinic`, `hospital`,
  `dentist`, `pharmacy`, `repair`, `salon`, `gym`, `school`, `laundry`, `service`. Список в
  prompt строится из enum, поэтому JSON schema и текстовая инструкция не расходятся.
- `pipeline/indexer.py:77-81`: VENUE теперь означает именованное физическое место, которое можно
  посетить для встреч, лечения, покупки лекарств, учёбы, спорта или очной услуги. Clinic и repair
  shop больше не исключаются самим определением.
- `storage/search.py:1234-1257`, `pipeline/indexer.py:275-312`, `index_cli.py:87-89,151-159`:
  добавлена bounded snapshot re-extraction по `no_venue` и `skipped`. Snapshot нужен потому, что
  повторный пустой ответ оставляет статус `no_venue`; live-query внутри цикла иначе повторно
  оплачивал бы одну строку до исчерпания `--limit`. `extracted` намеренно не разрешён: безопасный
  повтор такого статуса требует атомарно удалить или заменить старые `place_mentions`.

Команда 500-message pilot, запускать человеку в репозитории:

```bash
uv run python index_cli.py extract --status no_venue --limit 500
```

Команда использует существующий exclusive index lock, делает не более 500 сообщений и записывает
фактические `calls`, `messages`, `input_tokens`, `cached_input_tokens`, `output_tokens` в
`extraction_cost`. Live pilot в этой задаче не запускался.

## Pre-LLM filter

Источник `status='skipped'` найден в `SearchDatabase._seed_extraction_state()`.
До изменения единственным gate был `length(text) >= 80`: любое более короткое сообщение получало
`skipped`, даже `Lotus Dental clinic, запись к стоматологу` или
`MacLab Da Nang: ремонт MacBook`. Значит, риск для clinic/repair был реальным.

`storage/search.py:105-123,639-674` теперь оставляет общий length gate для длинных сообщений, но
пропускает короткие medical/service сообщения по EN/RU/VI vocabulary. То же правило переводит в
`pending` уже существующие подходящие строки со статусом `skipped` при следующем sync/build; затем
обычная crosspost deduplication не даёт платить за одинаковый текст несколько раз. Финальное
решение о наличии именованного физического места по-прежнему принимает extractor, а не regex.

## Tests

Новые tests в `tests/test_indexer.py:167-244` проверяют `clinic`, `repair`, `salon`, прежний `bar`,
targeted snapshot без повторной обработки `no_venue` и соблюдение `--limit`.
`tests/test_search.py:602-633` проверяет bypass и promotion старых `skipped` для коротких clinic,
repair и salon сообщений.

Во время полного gate обнаружены два старых независимых дефекта. Они исправлены минимально, чтобы
требование green suite было фактом, а не литературным жанром:

- `pipeline/crawler.py:189`: explicit `None` branch перед lookup anonymous sender исправляет
  существующую strict-mypy ошибку без изменения runtime semantics.
- `tests/test_recon_runner.py:73-81,135-140,310-321`: fixed-date fixture от 2026-07-01 вышла за
  30-day lookback; fake теперь использует текущее UTC-время и учитывает Telegram `offset_id`.

Финальная команда полного suite:

```text
$ .venv/bin/pytest --cov
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 35%]
........................................................................ [ 46%]
........................................................................ [ 58%]
........................................................................ [ 70%]
........................................................................ [ 81%]
........................................................................ [ 93%]
........................................                                 [100%]
TOTAL                         4249    441   1056    183    87%
Required test coverage of 80.0% reached. Total coverage: 86.58%
616 passed in 15.90s
```

Финальная mypy-команда и полный output:

```text
$ .venv/bin/mypy --strict
Success: no issues found in 38 source files
```

Дополнительный format/lint gate:

```text
$ .venv/bin/ruff format --check . && .venv/bin/ruff check .
70 files already formatted
All checks passed!
```

`uv run pytest --cov` сначала не запустил tests в sandbox, потому что попытался открыть global
cache вне repository:

```text
error: Failed to initialize cache at `/Users/nikitagorokhov/.cache/uv`
  Caused by: failed to open file `/Users/nikitagorokhov/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)
```

Поэтому проверки выполнены напрямую теми же executables из существующего locked `.venv`.

## Mutation check

Из `PlaceType` временно удалялось значение `CLINIC = "clinic"`. После мутации:

```text
$ .venv/bin/python -m py_compile pipeline/indexer.py
(no output, exit 0)
$ .venv/bin/pytest tests/test_indexer.py -q
F...........                                                             [100%]
FAILED tests/test_indexer.py::test_medical_service_and_existing_types_reach_the_place_index[...-Lotus Medical Clinic-clinic]
(exit 1; ровно clinic-case failed, остальные 11 tests passed)
```

После восстановления значения:

```text
$ .venv/bin/python -m py_compile pipeline/indexer.py
(no output, exit 0)
$ .venv/bin/pytest tests/test_indexer.py -q
............                                                             [100%]
```

## Token estimate для 31 232 `no_venue`

Схема `extraction_cost` находится в `storage/search_schema.sql:240-254` и хранит input, cached
input и output tokens отдельно. Экстраполяция 419 исторических сообщений:

```text
input/message  = 273129 / 419 = 651.859 tokens
output/message =  16458 / 419 =  39.279 tokens

input total  = 651.859 * 31232 = 20,358,866 tokens
output total =  39.279 * 31232 =  1,226,769 tokens
combined     =                       21,585,635 tokens
```

Ожидание: примерно **20.36M input tokens**, **1.23M output tokens**, **21.59M tokens total**.
Dollar estimate не указан: актуальную обычную и cached per-token цену `gpt-5.6-luna` нужно взять
у provider перед pilot/extrapolation.

## Regression round 2

Root cause находится не в taxonomy pre-filter и не в link extraction: fixed fixture создаёт все
обычные history messages с датой `2026-07-01` (`tests/test_recon_runner.py:73-81`), а
`ReconRunner` вычисляет настоящий 30-day cutoff относительно времени запуска
(`pipeline/recon.py:448-450`) и `TelegramCrawler` отбрасывает более старые сообщения до вызова
`extract_chat_links` (`pipeline/crawler.py:175-205`). На 2026-08-11 cutoff равен 2026-07-12,
поэтому сообщения не сохраняются и ссылки закономерно не попадают в следующую wave. Прямой probe
показал, что неизменённый extractor возвращает и `danang_villas`, и `danang_food`. Ослабить
production cutoff или считать окно от newest returned message означало бы нарушить контракт
`lookback_days`: dormant chat начал бы архивировать старые сообщения вне заказанного окна. Поэтому
по escape clause из fix spec production code и `tests/test_recon_runner.py` не изменялись; committed
test кодирует поведение, которое стало неверным после 2026-07-31. Отдельно, тот же untouched test
уже содержит wall-clock fixture в `tests/test_recon_runner.py:404-413`, хотя правила этого round
запрещают такую зависимость. Full-green output получить невозможно без изменения тестового clock
seam или без ухудшения production semantics, поэтому ниже записан фактический результат и работа
остановлена, как требует spec для неверного committed test.

Regression reproduction с untouched `tests/test_recon_runner.py`:

```text
$ .venv/bin/pytest tests/test_recon_runner.py -q
F.F..........                                                            [100%]
FAILED tests/test_recon_runner.py::test_a_topic_becomes_joined_chats_with_history
FAILED tests/test_recon_runner.py::test_links_in_history_become_next_wave_candidates
```

Прямой probe причины и link extractor:

```text
$ .venv/bin/python - <<'PY'
from datetime import UTC, datetime, timedelta
from pipeline.discovery import extract_chat_links
from pipeline.recon import _lookback_cutoff

for value in (
    "Villa for rent, see t.me/danang_villas",
    "also join t.me/danang_villas and @danang_food",
):
    print(value, [(x.username, x.invite_hash) for x in extract_chat_links(value)])
posted = datetime(2026, 7, 1, tzinfo=UTC)
cutoff = _lookback_cutoff(30)
print("posted", posted.isoformat())
print("cutoff", cutoff.isoformat())
print("discarded", posted < cutoff, "age_days", (datetime.now(UTC) - posted) / timedelta(days=1))
PY
Villa for rent, see t.me/danang_villas [('danang_villas', None)]
also join t.me/danang_villas and @danang_food [('danang_villas', None), ('danang_food', None)]
posted 2026-07-01T00:00:00+00:00
cutoff 2026-07-12T12:00:41.780476+00:00
discarded True age_days 41.500483570497686
```

Taxonomy tests, включая places-v2, medical/service types и targeted re-extraction:

```text
$ .venv/bin/pytest tests/test_indexer.py tests/test_search.py -o addopts='--strict-markers --timeout=15'
collected 142 items
tests/test_indexer.py ............                                       [  8%]
tests/test_search.py ................................................... [ 44%]
........................................................................ [ 95%]
.......                                                                  [100%]
============================= 142 passed in 3.29s ==============================
```

Полный suite, с точным остаточным статусом:

```text
$ .venv/bin/pytest --cov
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 35%]
........................................................................ [ 46%]
.............................................F.F........................ [ 58%]
........................................................................ [ 70%]
........................................................................ [ 81%]
........................................................................ [ 93%]
........................................                                 [100%]
Required test coverage of 80.0% reached. Total coverage: 86.22%
FAILED tests/test_recon_runner.py::test_a_topic_becomes_joined_chats_with_history
FAILED tests/test_recon_runner.py::test_links_in_history_become_next_wave_candidates
2 failed, 614 passed in 19.52s
```

Static gates:

```text
$ .venv/bin/mypy --strict
Success: no issues found in 38 source files

$ .venv/bin/ruff format --check . && .venv/bin/ruff check .
70 files already formatted
All checks passed!
```

## Round 3 — what the stale fixture was hiding (Claude, 2026-08-11)

The two failures blamed on the taxonomy change were **not** caused by it. Verified by stashing every
uncommitted change and running the file against clean `HEAD` (ddc7a4d): both fail there too.

Root cause: `_message()` pinned every fixture message to an absolute `datetime(2026, 7, 1, UTC)`,
while `_lookback_cutoff` (`pipeline/recon.py:448`) computes the window from `datetime.now()`. With
the default 30-day lookback the fixture fell out of the window on 2026-07-31 and the tests have been
red by calendar ever since. Fixed by making the default a single per-run value,
`FRESH_MESSAGE_DATE = now - 1 day`, deterministic within a run and always inside the window.
`datetime.now()` inline (the first attempt) would have worked but makes each message a different
instant, which is the kind of wall-clock dependence that produces intermittent tests.

`FakeTelegram` also ignored `offset_id` and replayed the same page forever, so "history ran out"
could never be observed. It now filters by `id < offset_id`, the way real pagination behaves.

**Open finding, deliberately left failing:** with those two repairs,
`test_backfill_stops_when_history_runs_out` fails with `history_calls == 2`, expected 1. It passed
before only because every fixture message was filtered out by date, so the runner saw an empty page
and stopped — green for the wrong reason. Now that the messages survive, the test measures what its
docstring claims ("a short page means there is nothing older to ask for") and shows the runner
issuing a second history request after a page shorter than the page size. Either the runner should
stop on a short page, or the test's expectation is wrong. Do not "fix" this by reverting the fixture
or relaxing the assertion — that restores a test that cannot fail for its stated reason. Needs a
decision on the intended backfill semantics.

Nothing here is committed. The taxonomy work (places-v2) is unaffected by any of it.

## Implementation round

### Что shipped

- Additive migration feature-detects все новые columns, создаёт `descriptors`,
  `place_descriptors`, `descriptor_embeddings` и `extraction_jobs`, backfill-ит временные
  `descriptor_text`/`offering_text` из legacy facets и проверяется через
  `PRAGMA foreign_key_check`. Существующий `place_fts` не удаляется и не переименовывается.
- `entities-v3` заменил закрытые `place_type`/`event_types` в provider schema на raw
  `descriptor`/`offerings`, `entity_kind` и `access_modes`. Старые facets теперь только
  детерминированная compatibility projection.
- Versioned job существует ровно один раз для `(corpus_id, prompt_version)`. Provider error меняет
  job, но не active snapshot. Success одной transaction заменяет mentions, пересчитывает
  descriptors/offerings/access modes/counters, удаляет orphan entity и только затем активирует
  `entities-v3`. Crossposts имеют свои jobs, но model получает только первую копию; остальные
  активируются локальным propagation.
- Category-dependent length gate удалён. Любой non-empty text с двумя Unicode letters или
  deterministic contact доходит до v3 job, поэтому `Барбер Дананг, пишите в личку @someone`
  больше не исчезает до extractor.
- `place_fts_next` является shadow trigram index по name, aliases, descriptor и offerings.
  Dual-compatible reader понимает `query`, `entity_kind` и `access_mode`; старый `name` остаётся
  identity constraint. Default reader всё ещё использует legacy `place_fts`.
- Offline gate содержит 48 queries: 42 verified на anonymized raw-message fixture и 6 hookah cases
  с явным `label_status=pending_evidence`. Runner считает recall@5, pooled precision@5, positive и
  negative non-empty share, name compatibility, critical cases и regression против committed
  baseline. CI использует frozen SQLite/NPZ artifacts и не вызывает provider.

### Identity и marketplace boundary

Provider без business name допустим только с contact, подтверждённым deterministic miner в том же
source message. Его canonical равен
`person:contact:<contact-kind>:<normalized-value>` (или такой же prefix с `organization`), поэтому
два барбера с разными handles не collision-ятся, а три написания одного handle не fragment-ят
entity. Proper name без contact использует `<kind>:name:<folded-name>:city:<folded-city>` только при
известном city. Поздний contact заменяет единственный точный name+city key; если contact key уже
занят другой entity, автоматического merge нет.

Граница marketplace проходит по устойчивости предложения: barber, cook, mover, repair person или
teacher продают ongoing service и входят в index; продажа одного подержанного телефона, короткая
аренда байка и разовая покупка товара не создают seller entity. Prompt содержит positive/negative
worked examples, а `entity-extraction-policy-v1.jsonl` фиксирует минимум два marketplace negatives.
Причина прозаическая: иначе 3 664 объявления создадут одноразовый seller landfill с хорошим recall
и нулевой пользовательской ценностью.

Hookah stratum не содержит выдуманный `place|sky lounge`. Все шесть cases pending до owner sweep;
если lounge не подтверждён raw evidence, они freeze-ятся как `must_be_empty`.

### Dark flags

```text
PLACE_EXPANDED_FTS_ENABLED=false
PLACE_SEMANTIC_ENABLED=false
PLACE_SEMANTIC_CUTOFF=0.55
```

Shadow FTS строится и синхронизируется при false flag, но production reader на него не переключён.
Semantic storage/query lane реализован для frozen vectors и имеет hard cutoff, однако query-time
provider embedding и lazy descriptor embedding worker не включены: cutoff ещё не calibrated, а
этот implementation round не тратит provider budget.

### Owner pilot

Команда гарантированно берёт bounded, round-robin sample из `extracted`, `no_venue` и `skipped` и
пишет composition в `selected_by_status`, а token receipt в `extraction_cost`:

```bash
uv run python index_cli.py extract \
  --status extracted \
  --status no_venue \
  --status skipped \
  --limit 500
```

Это paid extraction command; в implementation round она не запускалась. После pilot запустить
lexical smoke без provider calls:

```bash
uv run eidolon-place-eval --smoke --db data/eidolon_search.db --timeout 5
```

Reader switch оправдан только если одновременно выполнены условия:

1. Old/new name-query pool на candidate DB даёт 100% recall@5 и одинаковые entity keys для всех
   frozen name cases, включая обе формы SYNCHØUSE.
2. `place_fts_next` row count равен `places`, foreign keys чисты, а old index остаётся доступен для
   rollback.
3. Pilot review подтверждает person/service precision и marketplace exclusion; actual input/output
   tokens укладываются в согласованный budget.
4. Full candidate-DB gate проходит `recall@5 >= 0.90`, pooled precision `>= 0.80`, positive
   non-empty share `>= 0.95`, negative non-empty share `= 0`, без regression более `0.02`.
5. Hookah labels resolved raw evidence sweep-ом, а не результатом нового extractor.

До этого `PLACE_EXPANDED_FTS_ENABLED` и `PLACE_SEMANTIC_ENABLED` остаются false. Зелёный synthetic
fixture не является разрешением переключить production reader; SQLite тоже иногда умеет красиво
сдать экзамен, который сама себе написала.

### Verification и mutation kills

Offline retrieval gate прошёл 42 verified cases; 6 hookah cases честно остались
`pending_evidence`. Метрики fixture: macro recall@5 `1.00`, pooled precision@5 `1.00`, positive
non-empty `1.00`, negative non-empty `0.00`, name compatibility `1.00`. Lexical smoke прошёл все
deployment invariants за `0.0021s`; `PRAGMA integrity_check=ok`, foreign-key violations `[]`,
`places=8`, `place_fts_next=8`.

Четыре value mutations были убиты отдельными named tests, при этом module продолжал компилироваться,
а остальные tests соответствующего файла оставались зелёными:

- person fixture: `entity_kind=person` → `organization` убил
  `test_contact_identified_person_uses_contact_canonical_and_open_descriptor`;
- marketplace policy: `expected_entity_count=0` → `1` убил
  `test_marketplace_ads_are_negative_eval_cases_and_prompt_examples`;
- short self-promo: source text → `!` убил
  `test_short_self_promo_with_contact_reaches_entities_v3_job`;
- honest empty: zero query vector → repair vector убил
  `test_honest_empty_proctologist_case_returns_no_nearest_descriptor`.

Final static gates: `ruff format --check` reports 72 formatted files, `ruff check` passes, and
`mypy --strict` passes 39 source files. All tests except the pre-existing recon finding above pass.
The full suite still reports only
`test_backfill_stops_when_history_runs_out` (`history_calls=2`, expected `1`); this round does not
change crawler pagination semantics or the explicitly protected `tests/test_recon_runner.py`.

## Final measured defects: provider schema and author identity (2026-08-11)

Live verification was supplied by the owner; this implementation made no provider calls. Before
the schema fix, every extraction request failed with HTTP 400 because
`ExtractedEntity.access_modes` emitted unsupported `uniqueItems`. After the owner removed that
keyword, the same 10-message pack produced 10 entities, zero errors and one packed call. The Python
validator still enforces unique access modes. CI now recursively validates every Pydantic model
used as `response_format`: unsupported keywords, misplaced string/number/array constraints, a
non-object root, optional object properties, or missing `additionalProperties: false` fail before
deployment. Reintroducing `uniqueItems` failed both extraction schemas in the mutation probe.

The second measured defect was missing author context. For production `corpus_id=5279`, the barber
self-promotion body produced zero entities with only `Chat:` and `Date:`, but one person entity when
the header included `From: @barber_danang (Иван)`. There are 3,627 self-promotion messages in the
measured corpus and 4,483 messages whose author has an @handle. `entities-v4` therefore adds a
single-line `From:` header to packed and single-message extraction. `From:` is untrusted routing
metadata: it may provide a handle-first identity only when the body independently establishes an
ongoing service, and it can never be evidence. Ordinary chatter plus a named author stays empty.

The writer now accepts a deterministic contact from `sender_name` for person/organization identity;
otherwise a correct `@handle` returned by the model would be discarded after extraction. Telegram
ingestion and history capture persist `@handle (Display Name)` when both are available, and the
corpus name resolver upgrades older display-only rows once it observes a handle. Extraction
crosspost dedup now keys on body plus author identity: identical self-promo text from two different
people requires two answers. The version bump is required because a succeeded `entities-v3` job,
including the empty result for 5279, is immutable and would not otherwise be re-asked.

Owner-measured golden results folded into the regression evidence:

- `16455`: five organizations, all with descriptor/place type `hospital`;
- `111763`: person `@osteonavt`, descriptor `остеопат`;
- `124011`: person `Даша`, descriptor `остеопат`;
- `122743`: person `Надя Туйкина`, descriptor `остеопат`;
- `3596`: organization `citi dental`, descriptor `dental`, confidence `0.9`;
- `2603`, `2610`, `2611`: empty, preserving the one-off marketplace boundary;
- `5279`: empty without author metadata, person with the author header.

The exact short texts for `111763`, `124011`, and `3596` are all under 80 characters and are now
named regression cases for the category-neutral pre-LLM gate. The author policy dataset adds both
directions: self-promotion body plus author is a person, while weather chatter plus author is empty.
Changing the negative fixture to emit a person made the named integration test fail with two
entities instead of one, then the mutation was restored.

Final local verification used no paid calls. `ruff format --check` reports 73 formatted files,
`ruff check` and `mypy --strict` pass (`39 source files`). The full coverage suite reports
`643 passed`, total coverage `85.38%`, and only the protected pre-existing
`test_backfill_stops_when_history_runs_out` failure (`history_calls=2`, expected `1`). The frozen
place retrieval gate passes all 42 evaluated cases with macro recall@5 `1.0`, pooled precision@5
`1.0`, positive non-empty share `1.0`, negative non-empty share `0.0`, name compatibility `1.0`,
and six hookah cases still marked `pending_evidence`. `uv run` could not read the sandboxed global
uv cache, so the identical installed entrypoint was run as `.venv/bin/eidolon-place-eval`.

## Live pilot on production messages (Claude, 2026-08-11)

Run against a 409-message stratified sample exported read-only from production
(`corpus_messages`), extracted into a scratch DB. Production was not touched.

```
processed 394 · entities 53 · errors 0 · calls 20
input 70 478 · cached 28 766 · output 30 226 tokens · 94 seconds
entity_kind: person 21 · place 19 · organization 12
legacy place_type: other 21 · repair 17 · hospital 5 · cafe 3 · yoga 2 · …
descriptors seen: остеопат ×3, мастер маникюра ×2, мастер по бровям, мастермайнд,
                  коворкинг, экстатик dance, yoga trip, dinner theatre, спектакль
```

**The person layer is the headline.** 21 of 53 entities are people — a class the pipeline
could not represent at all before this round, and the one the corpus is richest in.
Descriptors like `мастер по бровям` and `мастермайнд` are exactly what no enum would ever
have contained.

Anchors (known-good production messages): 4 of 5 hit on the first run.

| corpus_id | expected | result |
|---|---|---|
| 16455 | 5 hospitals incl. `Bệnh viện Đa khoa Tâm Trí` | 5 mentions |
| 111763 | person `@osteonavt`, 33-char message | 1 mention |
| 124011 | person `Даша`, остеопат | 1 mention |
| 122743 | person `Надя Туйкина`, остеопат | 1 mention |
| 3596 | organization `citi dental`, 52-char message | missed on this run |
| 2603/2610/2611 | marketplace ads → nothing | 0, correct |

The `3596` miss was investigated rather than accepted: the pre-LLM gate passes it
(`is_extraction_candidate` → True) and the message was in the sample, so the miss happened
inside the model. Measured across pack sizes, it is found in 8 of 9 runs, and larger packs
are *better*, not worse (3/3 at 20, 2/3 at 2). Per-message extraction is therefore
stochastic at roughly 85-95% recall, and an anchor assertion must allow for that.

**Cost extrapolation to the full corpus**, from measured pilot rates
(179 input / 77 output tokens per message): ~12.0M input + ~5.1M output for all 66 944
messages, meaningfully below the 20.4M input estimated earlier from the v1 numbers. Cache
already absorbed 29% of input in the pilot and will do better on a long run.

Two defects were found by this pilot that no offline test caught, both now fixed:
`uniqueItems` in the response schema (every call 400d), and the author line missing from
the packed message (self-promotion invisible). Both are exactly the class of failure a
mocked test cannot see — the lesson is that at least one live path must be exercised
before an extractor is called done.

## Reply linkage

`reply_to_message_id` теперь проходит через live ingestion и reconnaissance capture/backfill в
`messages` / `scout_messages`, затем в `corpus_messages`. Значение читается из
`message.reply_to.reply_to_msg_id`: сам `MessageReplyHeader` не трактуется как integer. Все три
таблицы мигрируются feature detection без пересоздания. Повторный scout crawl обогащает уже
существующую строку reply link, но не считает её новым сообщением; последующий corpus sync
reconcile-ит строки, которые были проиндексированы до backfill.

Для answer-to-parent lookup используется `SearchDatabase.parent_text_for_reply()`. Non-reply и
reply с отсутствующим в corpus parent возвращают `None`; отсутствие parent является штатным. На
`corpus_messages` создаётся partial index `(chat_id, reply_to_message_id)` для сканирования answers
и measurement join. Parent lookup уже покрыт существующим UNIQUE index
`(chat_id, telegram_msg_id)`, отдельный parent index не нужен.

Bounded offline backfill live `raw_json` (повторять, пока `remaining` не станет `0`):

```bash
uv run python index_cli.py backfill-replies --limit 5000
```

Команда печатает `scanned`, `updated`, `no_reply`, `invalid_json` и `remaining`. Проверенные
non-reply строки помечаются отдельно от nullable reply id, поэтому следующий запуск двигается
дальше, а не перечитывает один и тот же prefix. Telegram и provider API команда не вызывает.
Строки без сохранённого `raw_json` восстановить offline невозможно.

После backfill или re-scrape сначала обновить derived corpus, затем измерить весь corpus:

```bash
uv run python index_cli.py sync
uv run python index_cli.py reply-stats
```

Или один chat:

```bash
uv run python index_cli.py reply-stats --chat-id -1001234567890
```

JSON явно называет population каждого числа: `reply_rows` считается от `rows_total`,
`replies_with_parent` от `reply_rows`, `replies_whose_parent_is_question` от
`replies_with_parent`. Последняя метрика использует только наличие `?` или full-width `？` в parent
text. Это слабое место плана: вопросы без знака вопроса будут undercounted, а риторические вопросы
попадут в count. Для первого viability measurement прозрачная deterministic эвристика лучше
скрытого LLM-вызова, но число нельзя называть semantic question recall. Исторический scout corpus
не получит links сам по себе: для него по-прежнему нужен re-scrape.

Offline verification: `mypy --strict` проверил 39 source files, `ruff format --check .` проверил
74 files, `ruff check .` прошёл. Full `pytest --cov` дал `652 passed`, total coverage `85.49%` и
единственный разрешённый failure
`test_backfill_stops_when_history_runs_out` (`history_calls=2`, expected `1`); test и pagination
semantics не менялись. Оба новых CLI subcommand отдельно прошли на temporary SQLite databases.

Четыре production-value mutation были проверены после `py_compile`; каждый named test стал red,
после восстановления все четыре снова green:

- crawler reply id `17` заменялся на `18` — падал
  `test_reply_field_survives_crawler_store_and_corpus_sync`;
- raw payload key `reply_to_msg_id` заменялся на `reply_to_msg_id_mutated` — падал
  `test_raw_json_reply_backfill_is_idempotent`;
- missing-parent fallback `None` заменялся на empty string — падал
  `test_reply_with_absent_parent_returns_none`;
- non-reply fallback `None` заменялся на empty string — падал `test_non_reply_returns_none`.
