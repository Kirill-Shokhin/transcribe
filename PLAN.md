# Transcribe Services — План

## Общая архитектура

Микросервисы с единым API стандартом. Каждый сервис — отдельный контейнер, включается через docker-compose profiles.

```
services/
├── asr/          # GigaAM-v3 + Silero VAD
├── llm/          # vLLM (GigaChat3 / Qwen3)
├── embed/        # embedding model (потом)
├── rag/          # structured RAG (потом)
├── gateway/      # FastAPI + HTML UI (потом)
└── redis/        # очередь, состояние
```

## Инфраструктура

- **GPU**: 3080 Ti (12GB VRAM)
- **Base image**: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- **Python**: 3.11+
- **PyTorch**: 2.5+
- **OS**: Windows + WSL2 Docker

## API стандарт (для всех сервисов)

```
GET  /health     # liveness probe
GET  /ready      # модель загружена и готова
POST /v1/{action}
```

Response format:
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

---

## Phase 1: ASR Service

### Модели
- **ASR**: GigaAM-v3 e2e_rnnt (HuggingFace, потом заменим на gigaam)
- **VAD**: Silero VAD (CPU)

### Конфигурация (env)
Все параметры в `config.py`, переопределяются через `ASR_*`:
```bash
ASR_REDIS_URL=redis://redis:6379/0
ASR_MODEL_NAME=ai-sage/GigaAM-v3
ASR_SHORT_AUDIO_THRESHOLD=30
ASR_WEBHOOK_TIMEOUT=60
```

### VRAM
- GigaAM: ~2GB
- VAD: CPU only

### Endpoints

```
POST /v1/transcribe
  Content-Type: multipart/form-data
  Body: file (audio)

Response:
{
  "success": true,
  "data": {
    "text": "полный текст",
    "segments": [
      {"start": 0.0, "end": 2.5, "text": "..."},
      ...
    ],
    "duration": 125.3,
    "language": "ru"
  }
}
```

### Текущий код

`transcribe.py` содержит рабочую логику:
- `merge_segments()` — склейка VAD сегментов в чанки до 25 сек
- `transcribe_long()` — основной пайплайн: load → VAD → chunk → ASR

**Изменения для сервиса:**
- ~~`AutoModel.from_pretrained()`~~ → `gigaam.load_model("v3_e2e_rnnt")`
- Убираем зависимость transformers
- `load_silero_vad()` остаётся

Логику переносим в `service.py`.

### Структура ASR сервиса

```
services/asr/
├── Dockerfile
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, lifespan
│   ├── config.py        # Settings (model paths, params)
│   ├── models.py        # Pydantic schemas
│   ├── service.py       # ASRService class (load models, transcribe)
│   └── routes/
│       ├── __init__.py
│       └── v1.py        # /v1/transcribe endpoint
└── tests/
```

---

## Phase 2: LLM Service

### Модели (выбор)
| Модель | VRAM (Q4) | Контекст | Комментарий |
|--------|-----------|----------|-------------|
| GigaChat3-10B-A1.8B | ~3GB | 131K | Native Russian, MoE |
| Qwen3-4B | ~2.5GB | 128K | Fallback, проверенная |

### Инференс
**vLLM** в Docker — OpenAI-compatible API из коробки.

```
POST /v1/chat/completions   # стандартный OpenAI формат
POST /v1/summarize          # обёртка для суммаризации
```

---

## Phase 3+: Embed, RAG, Gateway

- **Embed**: e5/bge multilingual
- **RAG**: qdrant + structured retrieval
- **Gateway**: FastAPI + HTML, роутинг на сервисы

---

## Docker Compose

```yaml
services:
  redis:
    image: redis:7-alpine

  asr:
    build: ./services/asr
    profiles: [asr, full]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  llm:
    image: vllm/vllm-openai:latest
    profiles: [llm, full]
    # ...

# Запуск: docker compose --profile asr up
```

---

## TODO

- [x] Определить архитектуру
- [x] Выбрать модели
- [x] Создать структуру services/asr/
- [x] Перенести логику из transcribe.py
- [x] Вынести константы в config.py
- [x] Dockerfile для ASR
- [x] docker-compose.yml
- [x] Заменить transformers на gigaam
- [ ] Тесты
- [ ] LLM сервис (Phase 2)
