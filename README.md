# FinAudit RL Environment (`finaudit-rl-env`)

Воспроизводимый симулятор и RL-среда в стиле Gymnasium для обучения и оценки моделей финансового ИИ-аудита (`GENESIS HACKATHON // FINTECH`).

Среда генерирует согласованную синтетическую бухгалтерию компании (главную книгу `ledger.csv`, счета `invoices.jsonl`, платежи `payments.jsonl`, курсы валют `fx.csv`, учётную политику `policy.json` и паспорт периода `manifest.json`), вносит причинные финансовые ошибки и правдоподобные чистые отвлекающие случаи (`distractors`), предоставляет узкий read-only интерфейс расследования с процессной изоляцией оценщика и оценивает выводы агента через открытый 1-to-1 матчер и устойчивую функцию награды.

---

## 1. Единая команда полной оффлайн-проверки

Проект работает **полностью автономно** (Python 3.11, без сети, без платных API и без GPU; укладывается в лимит 2 CPU, 4 ГБ RAM, < 60 сек и $\le 100$ шагов на эпизод).

### Локальный запуск (Windows / Linux / macOS)

```bash
# 1. Создание виртуального окружения Python 3.11 и установка зафиксированных зависимостей
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install --no-cache-dir -e .

# 2. ЕДИНАЯ КОМАНДА ПОЛНОЙ ПРОВЕРКИ ВСЕГО ПРОЕКТА
python -m finaudit_env.verify_all
```

Команда `python -m finaudit_env.verify_all` автоматически выполняет:
1. Полный прогон автотестов `pytest` (`70 passed`: схемы, генератор, контракт среды, процессная изоляция, матчер, устойчивость награды).
2. Независимую верификацию всех **20 открытых эпизодов** (`10 train` + `10 dev`) через `verify_episode` (баланс двойной записи до ошибки, причинное расхождение после и независимая достаточность `valid_evidence_sets`).
3. Прогон **4 базовых политик** (`random_explorer`, `simple_reconciler`, `flag_all_attack`, `oracle`) на всех 20 эпизодах + **20 парных проверок** воспроизводимости и изоморфной перестановки строк/ID (`permute_and_remap_episode`).
4. Сканирование записанных траекторий `artifacts/trajectories.jsonl` через `leak_checker` на отсутствие утечек `truth.json`, `provenance.json`, `ISS-` и путей файловой системы.

### Запуск в полностью изолированном Docker-контейнере (без сети)

```bash
docker build -t finaudit-rl-env .
docker run --rm --network none --cpus=2 --memory=4g finaudit-rl-env
```

---

## 2. Дополнительные CLI-команды

```bash
# Интерактивное демо работы среды и пошаговый разбор SimpleReconcilerPolicy
python -m finaudit_env.demo

# Пересборка 10 train и 10 dev эпизодов из версионированных конфигов configs/
python -m finaudit_env.build_datasets

# Запуск бенчмарка 4 базовых политик и генерация artifacts/trajectories.jsonl + artifacts/eval_report.json
python -m finaudit_env.run_benchmark

# Запуск всех unit- и интеграционных тестов
pytest -v
```

---

## 3. Структура репозитория

```text
finaudit-rl-env/
├── pyproject.toml                 # Зафиксированные зависимости (Python 3.11, pydantic==2.9.2, pytest==8.3.3)
├── Dockerfile                     # Полностью оффлайн-проверка в контейнере (--network none)
├── LICENSE                        # Лицензия MIT
├── README.md                      # Инструкция запуска и единая команда проверки
├── DESIGN.md                      # Записка о дизайне, метрики, разбор сбоев и ограничения
├── PROJECT_MAP.md                 # Справочник схем, типов ошибок и форматов ID
├── configs/                       # Версионированные конфигурации генерации (train_episodes.json, dev_episodes.json)
├── data/
│   ├── train/                     # 10 открытых train-эпизодов (ep_train_001 .. ep_train_010)
│   └── dev/                       # 10 открытых dev-эпизодов (ep_dev_001 .. ep_dev_010)
├── artifacts/
│   ├── trajectories.jsonl         # Записанные траектории без закрытого эталона в наблюдениях
│   └── eval_report.json           # Диагностика 4 политик (P, R, F1, reward, steps, time) + 20 парных проверок
├── src/finaudit_env/
│   ├── schemas.py                 # Строгие Pydantic v2 схемы (extra="forbid")
│   ├── generator/
│   │   ├── fx_utils.py            # Точная Decimal-конвертация валют с учётом разрядности и правил округления
│   │   ├── company.py             # Скрытая история чистой компании + distractors (split payments, recurring, refunds)
│   │   ├── errors.py              # Инъекция 4 типов ошибок + 2 сценариев взаимодействующих ошибок
│   │   ├── transforms.py          # Безвредные преобразования (перестановка строк + биективная смена ID)
│   │   └── verifier.py            # Независимый верификатор бухгалтерских равенств и доказательств
│   ├── env/
│   │   ├── datastore.py           # Read-only доступ к 6 открытым файлам + стабильная пагинация (<= 50 строк)
│   │   ├── calculator.py          # Безопасный AST-калькулятор на Decimal (без eval/кода/файлов)
│   │   ├── core.py                # RL-среда FinAuditEnv: reset/step, 5 действий, лимит 100 шагов и 60 сек
│   │   ├── evaluator_server.py    # Отдельный процесс оценщика (JSON-RPC по stdin/stdout)
│   │   ├── isolated_client.py     # Клиентский адаптер процессной изоляции (IsolatedFinAuditEnv)
│   │   └── leak_checker.py        # Сканер утечек эталона, секретных токенов и путей ФС
│   ├── scoring/
│   │   ├── matcher.py             # Открытый код нормализации ID, допусков и 1-to-1 двудольного матчинга
│   │   └── reward.py              # Расчёт P, R, F1 и терминальной награды clip(2*F1 - 1 - 0.002*steps, -1, 1)
│   ├── policies/
│   │   ├── base.py                # Общий интерфейс политик
│   │   ├── random_explorer.py     # Базовая политика 1: случайный исследователь с фиксированным seed
│   │   ├── simple_reconciler.py   # Базовая политика 2: простой алгоритм сверки через инструменты среды
│   │   ├── flag_all_attack.py     # Базовая политика 3: атака «пометить всё» (30 выводов)
│   │   └── oracle.py              # Базовая политика 4: Oracle (только для проверки максимума награды оценщика)
│   ├── build_datasets.py          # Генератор 10 train + 10 dev эпизодов
│   ├── run_benchmark.py           # Бенчмарк 4 политик на 20 эпизодах + 20 парных проверок
│   ├── demo.py                    # CLI-демо работы среды
│   └── verify_all.py              # Единая команда полной проверки всего проекта
└── tests/
    ├── test_schemas.py            # Тесты строгих схем данных и лимитов
    ├── test_generator.py          # Тесты генератора, баланса, ошибок, verifier и transforms
    ├── test_env_contract.py       # 20+ проверок контракта reset/step, лимитов, IPC-изоляции и отсутствия утечек
    └── test_scoring_reward.py     # 20+ проверок матчера, чистых компаний, дублей, атак и функции награды
```

---

## 4. Сводная диагностика базовых политик (20 эпизодов: 10 train + 10 dev)

| Базовая политика | Mean Precision | Mean Recall | Mean F1 | Mean Reward | Mean Steps | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `oracle` *(верхняя граница оценщика)* | `1.0000` | `1.0000` | `1.0000` | **`+0.9984`** | `1.80` | `0.0178` |
| `simple_reconciler` *(честная сверка)* | `0.9678` | `0.9875` | `0.9768` | **`+0.9314`** | `12.10` | `0.0655` |
| `random_explorer` *(случайный поиск)* | `0.2000` | `0.2000` | `0.2000` | **`-0.6029`** | `7.45` | `0.0234` |
| `flag_all_attack` *(атака «пометить всё»)* | `0.0000` | `0.0000` | `0.0000` | **`-1.0000`** | `3.00` | `0.0320` |

- **Парные проверки обобщения (`permute_and_remap_episode`)**: `20 / 20 PASS` (100% инвариантность ответа и награды `oracle` и `simple_reconciler` при полной смене ID и перестановке строк).
- **Проверка утечек эталона в `artifacts/trajectories.jsonl`**: `0 нарушений`.

Подробное описание архитектуры, алгоритма матчинга, разбора сбоев и ограничений приведено в [`DESIGN.md`](./DESIGN.md).
