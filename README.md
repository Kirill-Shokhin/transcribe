# Transcribe

ASR + LLM сервис для транскрибации аудио и суммаризации.

## Запуск

```bash
docker compose --profile full up -d
```

## Управление

```bash
# Запустить лёгкие сервисы (UI всегда доступен)
docker compose -p transcribe start gateway redis asr-api

# Остановить GPU-контейнеры вручную
docker compose -p transcribe stop llm asr-worker

# Остановить всё
docker compose -p transcribe down
```

## Порты

| Сервис | Порт |
|--------|------|
| UI | http://localhost:9000 |
| ASR API | http://localhost:9001 |
| LLM API | http://localhost:9002 |

## Архитектура

- **gateway** — UI + управление GPU-контейнерами
- **asr-api** — API + SQLite (jobs)
- **asr-worker** — GigaAM транскрибация (GPU, по требованию)
- **llm** — Qwen через llama.cpp (GPU, по требованию)
- **redis** — очередь задач

GPU-контейнеры автоматически:
- Стартуют при первом запросе
- Останавливаются через 2 мин idle
