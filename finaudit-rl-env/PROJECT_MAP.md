# PROJECT_MAP.md — Единый справочник контрактов и зон ответственности

Этот документ фиксирует все имена файлов, схемы данных, перечисления (`IssueType`), форматы идентификаторов и распределение модулей между участниками команды.

---

## 1. Правила параллельной работы в Git и разделение зон ответственности

> **Главное правило Git:** Не коммитить напрямую в ветку `main` при параллельной работе над модулями! Все изменения ведутся в изолированных feature-ветках и вливаются через Pull Request / согласованный merge после прохождения `pytest`.

| Роль | Git-ветка | Зона ответственности (папки и модули) | Описание задач |
| :--- | :--- | :--- | :--- |
| **Участник 1** | `feat/generator` | `src/finaudit_env/generator/`<br>`configs/`<br>`data/train/`, `data/dev/`<br>`src/finaudit_env/build_datasets.py` | Генерация чистой истории компании, курсы валют, инъекция 4 типов ошибок и взаимодействующих сценариев, перестановка/смена ID (`transforms.py`), проверка инвариантов (`verifier.py`), сборка 10 train и 10 dev эпизодов. |
| **Участник 2** | `feat/env-scoring` | `src/finaudit_env/env/`<br>`src/finaudit_env/scoring/`<br>`src/finaudit_env/policies/`<br>`src/finaudit_env/run_benchmark.py`<br>`src/finaudit_env/demo.py` | Хранилище открытых таблиц и пагинация (`datastore.py`), безопасный `calculator.py`, цикл `reset`/`step` (`core.py`), изоляция оценщика, `leak_checker.py`, 1-to-1 `matcher.py`, `reward.py` и 4 базовые политики. |
| **Общая зона** | Согласование обоих участников | `src/finaudit_env/schemas.py`<br>`src/finaudit_env/verify_all.py`<br>`tests/test_schemas.py`<br>`PROJECT_MAP.md` | Контракты Pydantic v2 (`schemas.py`) зафиксированы на старте. Любое изменение в `schemas.py` вносится **только по взаимному согласованию** обоих участников. |

---

## 2. Единая таблица всех 4 типов ошибок (`IssueType`)

Перечисление `IssueType` в `src/finaudit_env/schemas.py`:

| Значение `IssueType` (строка в JSON) | Имя в Python Enum | Описание финансовой ошибки | Типичные затрагиваемые объекты (`affected_ids`) | Минимальный допустимый набор доказательств (`evidence_ids`) |
| :--- | :--- | :--- | :--- | :--- |
| `"duplicate_invoice"` | `IssueType.DUPLICATE_INVOICE` | Дубликат счёта-фактуры (повторное проведение или выставление одного и того же счёта с тем же/искажённым номером или датой). | Канонические ID дублирующихся записей книги (`L-...`) и/или счетов (`INV-...`, `DOC-INV-...`). | Документы обоих счетов (`DOC-INV-...`) и соответствующие проводки в главной книге (`L-...`). |
| `"unallocated_or_partial_payment"` | `IssueType.UNALLOCATED_OR_PARTIAL_PAYMENT` | Пропущенное или частичное распределение платежа (платёж не привязан к счёту, либо сумма проводки/погашения не совпадает с суммой документа). | ID платежа (`PAY-...`, `DOC-PAY-...`), связанного счёта (`INV-...`) и проводки (`L-...`). | Платёжный документ (`DOC-PAY-...`), счёт (`DOC-INV-...`) и строки книги (`L-...`). |
| `"fx_conversion_error"` | `IssueType.FX_CONVERSION_ERROR` | Неверная дата или курс валютной конвертации (использован курс не на ту дату согласно `policy.json`, инвертированный курс или неверное округление). | ID проводки (`L-...`), первичного документа (`INV-...` / `PAY-...`) и строки курса (`FX-...`). | Проводка (`L-...`), документ в иностранной валюте (`DOC-INV-...` / `DOC-PAY-...`) и запись курса (`FX-YYYY-MM-DD-SRC-TGT`). |
| `"period_cutoff_error"` | `IssueType.PERIOD_CUTOFF_ERROR` | Ошибка отнесения к отчётному периоду (операция проведена в текущем периоде, хотя дата события относится к соседнему периоду согласно `policy.json`, или наоборот). | ID проводки (`L-...`) и первичного документа (`INV-...` / `DOC-INV-...` или `PAY-...` / `DOC-PAY-...`). | Проводка (`L-...`) и первичный документ с фактической датой операции (`DOC-INV-...` / `DOC-PAY-...`). |

---

## 3. Единые форматы идентификаторов (ID) и значений

Все идентификаторы в открытых таблицах, документах, эталоне (`truth.json`) и ответах агента (`findings.json`) подчиняются строгим форматам:

| Сущность | Формат ID | Пример | Где используется |
| :--- | :--- | :--- | :--- |
| **Запись главной книги (`LedgerRow`)** | `L-XXXX` (4+ цифры с ведущими нулями) | `L-0001`, `L-0142` | `ledger.csv` (`row_id`), `get_record(table="ledger", id="L-0001")`, `affected_ids`, `evidence_ids` |
| **Документ счёта (`InvoiceDoc`)** | `DOC-INV-XXX` (3+ цифры) | `DOC-INV-001` | `invoices.jsonl` (`doc_id`), `get_document(doc_id="DOC-INV-001")`, `evidence_ids`, `affected_ids` |
| **Бизнес-номер счёта (`invoice_id`)** | `INV-YYYY-XXX` | `INV-2026-001` | `invoices.jsonl` (`invoice_id`), `ledger.csv` (`invoice_id`), `payments.jsonl` (`invoice_refs`), `get_record(table="invoices", id="INV-2026-001")` |
| **Документ платежа (`PaymentDoc`)** | `DOC-PAY-XXX` (3+ цифры) | `DOC-PAY-001` | `payments.jsonl` (`doc_id`), `get_document(doc_id="DOC-PAY-001")`, `evidence_ids`, `affected_ids` |
| **Бизнес-номер платежа (`payment_id`)** | `PAY-YYYY-XXX` | `PAY-2026-001` | `payments.jsonl` (`payment_id`), `ledger.csv` (`payment_id`), `get_record(table="payments", id="PAY-2026-001")` |
| **Строка валютного курса (`FxRateRow`)** | `FX-YYYY-MM-DD-SRC-TGT` | `FX-2026-03-15-EUR-USD` | `fx.csv` (составной канонический ID строки курса), `get_record(table="fx", id="FX-2026-03-15-EUR-USD")`, `evidence_ids` |
| **Контрагент (`entity_id`)** | `ENT-XXX` | `ENT-001` | `ledger.csv`, `invoices.jsonl` |
| **Эталонная ошибка (`issue_id`)** | `ISS-XXX` | `ISS-001` | `truth.json` (`issue_id`, закрытый файл оценщика) |
| **Эпизод (`episode_id`)** | `ep_<split>_<NNN>` | `ep_train_001`, `ep_dev_001` | `manifest.json`, `truth.json`, `provenance.json`, `FindingsSubmission` |

### Правила представления данных:
- **Версия схемы**: `SCHEMA_VERSION = "1.0.0"`.
- **Валюты и разрядность (`CURRENCY_EXPONENTS`)**:
  `{"USD": 2, "EUR": 2, "GBP": 2, "KZT": 2, "JPY": 0, "CHF": 2}`.
- **Деньги**: Всегда целые минимальные единицы (`int`, `*_minor`, например центы для `USD` или тиыны для `KZT`, иены для `JPY`).
- **Курсы валют (`rate`)**: Десятичные строки с фиксированным форматом (например, `"1.0850"`), без `float` при сериализации.
- **Время**: Строки ISO-8601 с явным часовым поясом/смещением (например, `"2026-03-15T14:30:00+00:00"` или `"2026-03-15T19:30:00+05:00"`), даты курсов и границ периода — `"YYYY-MM-DD"`.
- **Отсутствующие ссылки**: Всегда явные `null` (`None` в Python) в JSON и пустая строка при сериализации `null`-полей в CSV (`invoice_id`, `payment_id` в `ledger.csv`).

---

## 4. Файловая структура одного эпизода (`data/<split>/<episode_id>/`)

| Имя файла | Видимость | Формат | Pydantic-модель в `schemas.py` | Назначение |
| :--- | :--- | :--- | :--- | :--- |
| `manifest.json` | Открытый (агенту) | JSON | `Manifest` | Метаданные эпизода: `schema_version`, `episode_id`, `base_currency`, `period_start`, `period_end`, `timezone`, счётчики записей/документов и правила округления. |
| `policy.json` | Открытый (агенту) | JSON | `PolicyConfig` | Учётная политика компании: правило выбора даты FX-курса, границы отчётного периода, режим округления и допуски. |
| `ledger.csv` | Открытый (через инструменты) | CSV | `LedgerRow` | Главная книга проводок (`row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id`). |
| `invoices.jsonl` | Открытый (через инструменты) | JSONL | `InvoiceDoc` (включая `LineItem`) | Первичные документы счетов-фактур. |
| `payments.jsonl` | Открытый (через инструменты) | JSONL | `PaymentDoc` | Первичные платёжные документы. |
| `fx.csv` | Открытый (через инструменты) | CSV | `FxRateRow` | Таблица курсов валют (`date,source_currency,target_currency,rate`). |
| `truth.json` | **ЗАКРЫТЫЙ (только оценщик)** | JSON | `TruthFile` (`list[TruthIssue]`) | Эталонные ошибки с `issue_id`, `type`, `affected_ids`, `expected_amount_minor`, `currency` и допустимыми наборами доказательств `valid_evidence_sets`. |
| `provenance.json` | **ЗАКРЫТЫЙ (только оценщик)** | JSON | `ProvenanceFile` (`list[ProvenanceEvent]`) | Полная история генерации, seed, базовых операций и внесённых причинных искажений. |

---

## 5. Контракт действий агента (`Action`) и отправки ответа (`FindingsSubmission`)

Агент взаимодействует со средой через JSON-совместимый Discriminated Union `Action` (поле-дискриминатор `action`):

1. `list_records`: `{"action": "list_records", "table": "ledger" | "invoices" | "payments" | "fx", "cursor": null | "0", "limit": 50}` (максимум 50 строк на страницу).
2. `get_record`: `{"action": "get_record", "table": "ledger" | "invoices" | "payments" | "fx", "id": "L-0001"}`.
3. `get_document`: `{"action": "get_document", "doc_id": "DOC-INV-001"}`.
4. `calculator`: `{"action": "calculator", "expression": "(125000 * 1.085) - 135625"}`.
5. `submit`: `{"action": "submit", "submission": {"episode_id": "...", "findings": [...]}}`.

Лимиты `FindingsSubmission` и среды:
- Максимум **100 шагов** на эпизод (включая `submit`).
- Максимум **30 выводов (`findings`)** в одной отправке.
- Максимум **10 ID доказательств (`evidence_ids`)** на один вывод.
- Максимум **1 000 символов** в поле `explanation` на один вывод.
