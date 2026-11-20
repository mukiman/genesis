# FinAudit RL Environment (`finaudit-rl-env`)

Воспроизводимый симулятор и RL-среда для финансового ИИ-аудитора: синтетические финансовые документы компании, изолированные инструменты расследования, скрытые финансовые ошибки и функция награды.

---

## Быстрый старт и установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/mukiman/genesis.git
cd genesis/finaudit-rl-env
```

### 2. Создание виртуального окружения (Python 3.11)

С помощью `uv` (рекомендуется):
```bash
uv venv --python 3.11 .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
uv pip install -e .
```

Либо стандартным `venv` + `pip`:
```bash
python3.11 -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
pip install -e .
```

### 3. Проверка контрактов и запуск тестов

```bash
pytest tests/test_schemas.py
```

---

## Правила командной работы в Git (Параллельная разработка)

Чтобы исключить конфликты при параллельной разработке, **не коммитьте напрямую в `main`** во время активной фазы разработки модулей.

1. **Участник 1 (Генератор данных, эталон и датасеты)** — работает в ветке `feat/generator`:
   ```bash
   git checkout -b feat/generator
   ```
   - **Зона ответственности**: `src/finaudit_env/generator/`, `configs/`, `data/`, `src/finaudit_env/build_datasets.py`.

2. **Участник 2 (RL-среда, изоляция, скоринг и базовые политики)** — работает в ветке `feat/env-scoring`:
   ```bash
   git checkout -b feat/env-scoring
   ```
   - **Зона ответственности**: `src/finaudit_env/env/`, `src/finaudit_env/scoring/`, `src/finaudit_env/policies/`, `src/finaudit_env/run_benchmark.py`, `src/finaudit_env/demo.py`.

3. **Общий контракт (`src/finaudit_env/schemas.py` и `PROJECT_MAP.md`)**:
   - Изменяется **только по взаимному согласованию** обоих участников и сразу синхронизируется между ветками.

Подробное описание форматов ID, типов ошибок, структуры файлов эпизода и контрактов см. в [`PROJECT_MAP.md`](./PROJECT_MAP.md) и [`DESIGN.md`](./DESIGN.md).
