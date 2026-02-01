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
- **ASR**: GigaAM-v3 e2e_rnnt (HuggingFace: `ai-sage/GigaAM-v3`, revision: `e2e_rnnt`)
- **VAD**: Silero VAD (CPU)

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
- Загрузка моделей через `AutoModel.from_pretrained()` и `load_silero_vad()`

Эту логику переносим в `service.py`.

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
- [ ] Создать структуру services/asr/
- [ ] Перенести логику из transcribe.py
- [ ] Dockerfile для ASR
- [ ] docker-compose.yml
- [ ] Тесты
- [ ] LLM сервис (Phase 2)
