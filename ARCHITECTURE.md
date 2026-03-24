# DnS Trading RAG 챗봇 — 시스템 아키텍처

> 데이터 사이언티스트를 위한 아키텍처 가이드.
> 코드의 구조, 데이터 흐름, 설계 패턴을 실제 코드와 함께 설명합니다.

---

## 1. 시스템 개요

이 프로젝트는 **RAG (Retrieval-Augmented Generation)** 기반 카카오톡 챗봇입니다.
Dropbox 파일과 네이버 메일을 벡터 DB에 인덱싱하고, 사용자 질문에 대해
관련 문서를 검색한 뒤 LLM이 답변을 생성합니다.

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ① 데이터 수집 (Ingestion)         ② 벡터 DB (Storage)            │
│  ┌──────────┐  ┌──────────┐       ┌─────────────────────┐        │
│  │ Dropbox  │  │ 네이버   │       │   Zilliz Cloud      │        │
│  │  API     │  │ IMAP     │       │   (Milvus)          │        │
│  └────┬─────┘  └────┬─────┘       │                     │        │
│       │              │             │  documents (벡터)    │        │
│       ▼              ▼             │  sync_state (커서)   │        │
│  ┌─────────────────────────┐      │  briefings (브리핑)   │        │
│  │ 텍스트 추출 → 청킹       │──▶  │  chat_logs (로그)    │        │
│  │ → 임베딩 → 인덱싱       │      └──────────┬──────────┘        │
│  └─────────────────────────┘                 │                    │
│       ▲ 30분마다 (GitHub Actions)             │                    │
│                                              ▼                    │
│  ③ RAG 파이프라인             ④ 서빙 (Serving)                    │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │ 질문 임베딩           │    │ FastAPI (Render)    │              │
│  │ → 벡터 검색 (top-5)  │◀──│  /skill/query       │◀── 카카오톡  │
│  │ → LLM 답변 생성      │──▶│  /skill/briefing    │──▶ 사용자    │
│  │   (Gemini Flash)     │    │  /admin/logs        │              │
│  └─────────────────────┘    └─────────────────────┘              │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 핵심 기술 스택

| 역할 | 기술 | 왜 이것을 선택했는가 |
|------|------|---------------------|
| 벡터 DB | Zilliz Cloud (Milvus) | 무료 5GB, 벡터 유사도 검색 전문 |
| 임베딩 | Gemini `gemini-embedding-001` | 768차원, API 기반 (GPU 불필요) |
| LLM | Gemini 3 Flash (폴백: 2.5 Flash) | 저렴 ($0.50/$3.00 per 1M 토큰) |
| 웹 서버 | FastAPI on Render | 비동기(async) 지원, 무료 호스팅 |
| 스케줄링 | GitHub Actions | 무료 CI/CD, cron 스케줄 지원 |
| 챗봇 | 카카오톡 오픈빌더 | 카카오 스킬 서버 연동 |

---

## 2. 프로젝트 구조

```
project-root/
├── src/                          # 프로덕션 코드 (4,600+ 줄)
│   ├── config.py                 # 환경 변수 관리
│   ├── db/
│   │   └── zilliz_client.py      # DB 연결 + 컬렉션 초기화
│   ├── ingestion/                # 데이터 수집 파이프라인
│   │   ├── text_extractor.py     # 파일 형식별 텍스트 추출
│   │   ├── chunker.py            # 텍스트 → 청크 분할
│   │   ├── indexer.py            # 임베딩 + DB 저장
│   │   ├── dropbox_sync.py       # Dropbox 증분 동기화
│   │   └── naver_mail_sync.py    # 네이버 메일 IMAP 동기화
│   ├── rag/                      # RAG 파이프라인
│   │   ├── embedder.py           # 텍스트 → 벡터 변환
│   │   ├── retriever.py          # 벡터 유사도 검색
│   │   ├── generator.py          # LLM 답변 생성
│   │   └── chain.py              # RAG 오케스트레이션
│   ├── server/                   # API 서버
│   │   ├── main.py               # FastAPI 진입점
│   │   ├── skill_handler.py      # 카카오톡 스킬 엔드포인트
│   │   ├── callback.py           # 비동기 콜백 처리
│   │   ├── chat_logger.py        # 대화 로깅 + 비용 추적
│   │   └── admin.py              # 관리자 API
│   └── briefing/                 # 브리핑 기능
│       ├── generator.py          # 브리핑 생성 (6종)
│       └── sender.py             # 브리핑 저장/발송
├── scripts/                      # 유틸리티 스크립트
│   ├── full_sync.py              # Dropbox + 메일 동기화 실행
│   ├── run_briefing.py           # 브리핑 수동 실행
│   ├── ocr_reprocess.py          # PDF OCR 재처리
│   └── reprocess_missing.py      # 비PDF 파일 일괄 처리
└── .github/workflows/            # 자동화
    ├── sync_data.yml             # 30분 주기 데이터 동기화
    └── daily_briefing.yml        # 평일 19시 브리핑 생성
```

---

## 3. 레이어별 상세 설명

### 3.1 설정 관리 (`src/config.py`)

Pydantic의 `BaseSettings`로 환경 변수를 타입 안전하게 관리합니다.
`.env` 파일이나 OS 환경 변수에서 자동으로 값을 읽어옵니다.

```python
class Settings(BaseSettings):
    zilliz_uri: str = ""           # 빈 문자열 = 기본값 (설정 안 하면 에러)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_fallback_model: str = "gemini-2.5-flash"
    embedding_dim: int = 768
    chunk_size: int = 500
    chunk_overlap: int = 50
    # ...
    model_config = {"env_file": ".env"}  # .env 파일에서 자동 로드
```

**설계 포인트**: 모든 설정을 한 곳에서 관리하고, 환경(로컬/서버)에 따라
`.env` 또는 시스템 환경 변수로 유연하게 주입합니다.

---

### 3.2 데이터베이스 레이어 (`src/db/zilliz_client.py`)

#### 싱글턴(Singleton) 패턴

```python
_client: MilvusClient | None = None

def get_client() -> MilvusClient:
    global _client
    if _client is not None:
        return _client          # 이미 생성됨 → 재사용
    _client = MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)
    return _client
```

**왜 싱글턴인가?** DB 커넥션은 생성 비용이 높습니다. 매번 새로 만들면 느리고
리소스를 낭비합니다. 한 번 만들어서 모든 모듈이 공유합니다.

#### 4개 컬렉션 (테이블)

| 컬렉션 | 용도 | 핵심 필드 |
|--------|------|----------|
| `documents` | 문서 청크 + 벡터 | embedding(768d), content, source_type, filename |
| `sync_state` | 동기화 커서 저장 | sync_type, last_cursor, last_sync_time |
| `briefings` | 생성된 브리핑 저장 | briefing_type, content, generated_at |
| `chat_logs` | 대화 로그 + 비용 | user_query, response, cost_usd, tokens |

> **참고**: Milvus는 벡터 DB라서 모든 컬렉션에 벡터 필드가 필수입니다.
> 벡터 검색이 불필요한 컬렉션에는 `_dummy_vec`(2차원)을 넣어 제약을 충족합니다.

---

### 3.3 데이터 수집 파이프라인 (Ingestion)

데이터가 DB에 저장되기까지 4단계를 거칩니다:

```
원본 파일/메일 → [텍스트 추출] → [청킹] → [임베딩] → [인덱싱(DB 저장)]
```

#### 3.3.1 텍스트 추출 (`text_extractor.py`) — 디스패치 패턴

파일 확장자에 따라 적절한 추출 함수를 자동 선택합니다.

```python
# 디스패치 테이블: 확장자 → 추출 함수 매핑
_EXTRACTOR_MAP: dict[str, ExtractorFn] = {
    ".pdf":  _extract_pdf,     # PyMuPDF → OCR 폴백
    ".docx": _extract_docx,    # python-docx
    ".xlsx": _extract_xlsx,    # openpyxl
    ".hwp":  _extract_hwp,     # gethwp → pyhwp 폴백
    ".zip":  _extract_zip,     # 재귀 해제
    # ...
}

def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    extractor = _EXTRACTOR_MAP.get(ext)
    if extractor is None:
        return ""
    return extractor(file_path)  # 확장자에 맞는 함수 호출
```

**설계 포인트**: if-elif 체인 대신 딕셔너리 디스패치를 사용하면
새 파일 형식 추가가 간단합니다 (함수 하나 작성 → 테이블에 등록).

**OCR 전략 (PDF)**:
```
1차: PyMuPDF 텍스트 추출 (텍스트 PDF → 즉시 성공)
2차: Apple Vision OCR (macOS에서만, 빠르고 정확)
3차: PaddleOCR (Linux/GitHub Actions, 다국어 지원)
```

#### 3.3.2 텍스트 청킹 (`chunker.py`) — 재귀 분할

긴 문서를 겹치는 작은 조각(청크)으로 나눕니다.

```python
class TextChunker:
    def __init__(self):
        self.chunk_size = 500      # 한 청크 최대 500자
        self.chunk_overlap = 50    # 연속 청크 간 50자 겹침
        self.separators = ["\n\n", "\n", ". ", " ", ""]  # 분할 우선순위
```

**왜 청킹하는가?**
- LLM에는 컨텍스트 길이 제한이 있습니다
- 벡터 검색은 짧은 텍스트에서 더 정확합니다
- 관련 없는 내용까지 LLM에 넘기면 답변 품질이 떨어집니다

**왜 오버랩이 있는가?**
문장이 청크 경계에서 잘리면 의미를 잃습니다.
50자 겹침으로 경계 부분의 문맥을 보존합니다.

```
원본: "A사와의 계약 기한은 3월 15일이며 납품 수량은 100대입니다."
청크1: "A사와의 계약 기한은 3월 15일이며 납품"  (500자)
청크2: "3월 15일이며 납품 수량은 100대입니다."  (겹침 ← "3월 15일이며 납품")
```

**재귀 분할 알고리즘**:
```
1. "\n\n" (단락 경계)으로 나눠봄 → 500자 이하면 OK
2. 너무 크면 "\n" (줄바꿈)으로 재시도
3. 아직도 크면 ". " (문장 경계)으로 재시도
4. 그래도 크면 " " (단어 경계)으로 재시도
5. 최후 수단: 글자 단위로 자름
```

**출력 형태**:
```python
@dataclass
class Chunk:
    text: str          # 청크 텍스트
    chunk_index: int   # 0, 1, 2, ... (원본 문서 내 순서)
    metadata: dict     # 원본 문서의 메타데이터 복사본
```

#### 3.3.3 임베딩 (`rag/embedder.py`) — 텍스트를 벡터로

텍스트를 768차원 실수 벡터로 변환합니다.
의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치합니다.

```python
class Embedder:
    _instance = None  # 싱글턴

    def embed(self, text: str) -> list[float]:
        """단일 텍스트 → 768차원 벡터"""
        return self._call_embed_api(text)[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 100개씩 배치로 임베딩"""
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            vectors = self._call_embed_api(batch)
            # ...
```

**Rate Limiting 전략**:
```python
def _call_embed_api(self, contents):
    for attempt in range(8):          # 최대 8회 재시도
        self._pace()                   # API 호출 간 1초 간격
        try:
            result = self._client.models.embed_content(...)
            return result.embeddings
        except Exception as e:
            if "429" in str(e):        # 요청 초과
                wait = min(2 ** attempt * 5, 65)  # 지수 백오프
                time.sleep(wait)       # 5, 10, 20, 40, 65초 대기
```

**지수 백오프(Exponential Backoff)**:
API가 "너무 많은 요청"(429)을 반환하면, 대기 시간을 점점 늘립니다.
`2^0 × 5 = 5초`, `2^1 × 5 = 10초`, `2^2 × 5 = 20초` ...
이렇게 하면 API 서버에 부담을 주지 않으면서 안정적으로 재시도합니다.

#### 3.3.4 인덱싱 (`ingestion/indexer.py`) — DB에 저장

청크들을 임베딩하고 Zilliz에 upsert합니다.

```python
class Indexer:
    def index_document(self, chunks, metadata) -> int:
        # 1. 기존 데이터 삭제 (같은 source_id)
        self.delete_document(metadata.source_id)

        # 2. 모든 청크 임베딩
        embeddings = self._embedder.embed_batch([c.text for c in chunks])

        # 3. 행(row) 생성
        rows = self._build_rows(chunks, embeddings, metadata)

        # 4. 배치 삽입 (100개씩)
        return self._batch_insert(rows)
```

**Delete-then-Insert 전략**:
문서가 수정되면 이전 청크를 모두 삭제하고 새로 삽입합니다.
이렇게 하면 오래된 데이터가 남지 않습니다.

**에러 복구**: 배치 삽입이 실패하면 개별 행 단위로 재시도합니다.
```python
def _batch_insert(self, rows):
    try:
        self._client.insert(data=batch)       # 100개 한 번에
    except Exception:
        self._insert_individually(batch)       # 실패 시 1개씩
```

#### 3.3.5 Dropbox 동기화 (`dropbox_sync.py`) — 커서 기반 증분 동기화

```
[첫 실행]
    files_list_folder("/DnS 사업관리")  →  전체 파일 목록 + 커서(cursor)
    커서를 sync_state에 저장

[이후 실행]
    files_list_folder_continue(커서)  →  변경된 파일만
    ├─ FileMetadata (추가/수정) → 다운로드 → 추출 → 청킹 → 인덱싱
    ├─ DeletedMetadata (삭제)   → DB에서 해당 source_id 삭제
    └─ 새 커서 저장
```

**커서(Cursor)란?** Dropbox가 발행하는 "여기까지 봤습니다" 마커입니다.
다음 호출 시 커서를 보내면 그 이후 변경분만 반환합니다.
전체 스캔 대비 API 호출과 처리 시간을 크게 줄입니다.

#### 3.3.6 네이버 메일 동기화 (`naver_mail_sync.py`) — IMAP SINCE 쿼리

```python
# IMAP 프로토콜로 메일 서버에 직접 연결
mailbox = MailBox("imap.naver.com").login(email, password)

# 마지막 동기화 이후 메일만 가져오기
for msg in mailbox.fetch(AND(date_gte=last_sync_date)):
    # 본문 처리
    body = BeautifulSoup(msg.html, "html.parser").get_text()
    # 첨부파일 처리
    for att in msg.attachments:
        text = extract_text(att)
    # 청킹 → 인덱싱
```

**동기화 대상 폴더**: INBOX, Sent, CAS, SOI, Nanotech, DAPA, RKCC 등 14개

---

### 3.4 RAG 파이프라인

RAG의 핵심 아이디어: **LLM이 모르는 정보를 외부 문서에서 찾아서 같이 보내준다.**

```
사용자 질문: "A사 납품 기한?"
        │
        ▼
 ┌─ Retriever ─────────────────────────────────────┐
 │  1. 질문을 768차원 벡터로 변환                      │
 │  2. DB에서 코사인 유사도로 가장 가까운 문서 5개 검색   │
 │  3. 검색 결과를 LLM 프롬프트용 텍스트로 포맷          │
 └──────────────────────────┬──────────────────────┘
                            │
                            ▼
 ┌─ Generator ─────────────────────────────────────┐
 │  시스템 프롬프트:                                  │
 │    "문서 컨텍스트만으로 답변하세요. 없으면 모른다고."   │
 │                                                   │
 │  유저 프롬프트:                                    │
 │    "검색 결과: [문서1] A사 납품계약 ... 3월 15일     │
 │     질문: A사 납품 기한?"                          │
 │                                                   │
 │  → Gemini 3 Flash 호출                            │
 │  → "A사 납품 기한은 2025년 3월 15일입니다."          │
 └──────────────────────────┬──────────────────────┘
                            │
                            ▼
                    사용자에게 답변 전달
```

#### 3.4.1 벡터 검색 (`retriever.py`)

```python
class Retriever:
    def search(self, query, source_type=None, top_k=5):
        # 1. 질문을 벡터로 변환
        query_embedding = self.embedder.embed(query)

        # 2. 필터 구성 (선택사항)
        filter_expr = ""
        if source_type:
            filter_expr = f'source_type == "{source_type}"'

        # 3. Milvus ANN(근사 최근접 이웃) 검색
        results = self.client.search(
            collection_name="documents",
            data=[query_embedding],
            filter=filter_expr,
            limit=top_k,
            search_params={"metric_type": "COSINE"},
        )
        return results
```

**코사인 유사도(COSINE)**: 두 벡터의 방향이 얼마나 같은지 측정합니다.
값이 1에 가까울수록 의미가 비슷합니다.

```
"납품 기한 알려줘" ←→ "A사 납품계약서 ... 기한 3/15"  = 유사도 0.85 ✓
"납품 기한 알려줘" ←→ "회의록 ... 점심 메뉴 논의"     = 유사도 0.12 ✗
```

#### 3.4.2 LLM 답변 생성 (`generator.py`) — 폴백 패턴

```python
class Generator:
    async def _call_with_fallback(self, prompt) -> tuple[str, dict]:
        try:
            return await self._call_llm(self.model, prompt)         # 1차: Gemini 3 Flash
        except Exception:
            pass
        try:
            return await self._call_llm(self.fallback_model, prompt) # 2차: Gemini 2.5 Flash
        except Exception:
            return "답변을 생성할 수 없습니다.", {}                   # 3차: 에러 메시지
```

**폴백(Fallback) 패턴**: 주 모델이 실패하면 (API 오류, 과부하 등)
자동으로 대체 모델을 시도합니다. 사용자 경험이 끊기지 않습니다.

**토큰 사용량 추적**:
```python
async def _call_llm(self, model, prompt):
    response = await self.client.aio.models.generate_content(
        model=model, contents=prompt, config=config
    )

    # 응답에서 토큰 수 추출
    usage = {"model": model, "input_tokens": 0, "output_tokens": 0}
    if response.usage_metadata:
        usage["input_tokens"] = response.usage_metadata.prompt_token_count
        usage["output_tokens"] = response.usage_metadata.candidates_token_count

    return response.text, usage  # 답변 텍스트 + 사용량 메타데이터
```

#### 3.4.3 RAG 체인 (`chain.py`) — 오케스트레이션

Retriever와 Generator를 하나의 파이프라인으로 연결합니다.

```python
class RAGChain:
    async def run(self, query: str) -> str:
        t_start = time.monotonic()

        # Step 1: 검색 (동기 함수라 스레드에서 실행)
        results, context, sources = await asyncio.to_thread(
            self.retriever.search_and_prepare, query
        )

        if not results:
            return "관련 문서를 찾을 수 없습니다."

        # Step 2: LLM 답변 생성
        answer, usage = await self.generator.generate(query, context, sources)

        # Step 3: 카카오톡 1000자 제한
        answer = _truncate(answer, 1000)

        # Step 4: 비동기 로깅 (응답 지연 없음)
        asyncio.create_task(chat_logger.log_chat(
            query_type="rag", user_query=query,
            response=answer, usage=usage,
            response_time_ms=int((time.monotonic() - t_start) * 1000),
        ))

        return answer
```

**`asyncio.to_thread()`**: 동기(sync) 함수를 비동기(async) 환경에서 실행합니다.
벡터 검색은 동기 함수지만 이벤트 루프를 블로킹하지 않도록 별도 스레드에서 실행합니다.

**`asyncio.create_task()`** (Fire-and-Forget):
로깅은 사용자 응답에 영향을 주면 안 됩니다.
`create_task()`로 백그라운드에서 실행하고 결과를 기다리지 않습니다.

---

### 3.5 서빙 레이어 (API 서버)

#### 3.5.1 카카오톡 5초 타임아웃 해결 — 콜백 패턴

카카오톡 스킬 서버는 **5초 안에 응답**해야 합니다.
RAG 파이프라인은 보통 5~15초 걸립니다. 어떻게 해결할까요?

```
시간   카카오톡            서버                    LLM
 0s    질문 전송 ──────▶  요청 수신
 0.1s                     즉시 응답 반환 ──────▶  "🔍 검색 중..."
 0.2s                     백그라운드 태스크 시작
 3s                       벡터 검색 완료
 8s                       LLM 답변 생성 완료
 8.1s                     callbackUrl로 POST ──▶  최종 답변 표시
```

```python
# src/server/skill_handler.py
@router.post("/skill/query")
async def skill_query(request: Request):
    body = await request.json()
    utterance = body["userRequest"]["utterance"]
    callback_url = body["userRequest"].get("callbackUrl")

    if callback_url:
        # 비동기: 즉시 응답 + 백그라운드 처리
        asyncio.create_task(
            process_and_callback(utterance, callback_url)
        )
        return {
            "version": "2.0",
            "useCallback": True,  # ← "나중에 결과 보낼게"
            "template": {"outputs": [
                {"simpleText": {"text": "문서를 검색하고 있습니다..."}}
            ]}
        }
    else:
        # 동기: 5초 안에 빠른 답변 시도
        answer = await asyncio.wait_for(chain.quick_run(utterance), timeout=4.5)
        return {"version": "2.0", "template": {"outputs": [
            {"simpleText": {"text": answer[:1000]}}
        ]}}
```

**콜백 URL 제약**:
- 유효 시간: 1분
- 사용 횟수: 1회
- 봇테스터에서 테스트 불가 (실제 카카오톡에서만)

#### 3.5.2 긴 응답 분할 (`callback.py`)

카카오톡은 한 말풍선에 1000자 제한이 있습니다.
긴 브리핑은 최대 3개 말풍선으로 분할합니다.

```python
def _split_text_for_kakao(text, max_chars=1000, max_outputs=3):
    if len(text) <= max_chars:
        return [text]           # 짧으면 그대로

    chunks = []
    remaining = text
    while remaining and len(chunks) < max_outputs:
        # 줄바꿈 위치에서 자르기 (문장 중간 절단 방지)
        cut = remaining.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars     # 줄바꿈 없으면 하드 컷

        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks  # ["말풍선1", "말풍선2", "말풍선3"]
```

#### 3.5.3 브리핑 자동 감지 (`skill_handler.py`)

사용자 발화에서 브리핑 키워드를 패턴 매칭으로 감지합니다.

```python
_BRIEFING_TYPE_PATTERNS = [
    (("어제 업무", "어제 브리핑", "어제 한 일"), "yesterday"),
    (("지난 주", "지난주", "저번 주"),           "last_week"),
    (("오늘 업무", "오늘 브리핑", "일간"),        "daily"),
    (("이번 주", "이번주", "주간"),              "weekly"),
    # ...
]

def _detect_briefing_request(utterance):
    for keywords, btype in _BRIEFING_TYPE_PATTERNS:
        for kw in keywords:
            if kw in utterance:
                return btype       # "오늘 브리핑" → "daily"
    return None                    # 브리핑 아님 → RAG로 라우팅
```

**순서가 중요합니다**: "어제"를 먼저 체크한 뒤 "오늘"을 체크합니다.
"지난 주"를 먼저 체크한 뒤 "이번 주"를 체크합니다.
(더 구체적인 패턴을 먼저 매칭)

---

### 3.6 비용 추적 (`chat_logger.py`)

모든 LLM 호출의 토큰 수와 비용을 자동으로 기록합니다.

```python
_MODEL_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # $/1M 토큰
    "gemini-2.5-flash":       {"input": 0.15, "output": 0.60},
}

def _calculate_cost(model, input_tokens, output_tokens):
    pricing = _MODEL_PRICING.get(model, {"input": 0.50, "output": 3.00})
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
```

**예시**: 300 입력 토큰 + 134 출력 토큰 (Gemini 3 Flash)
```
비용 = 300 × $0.50/1M + 134 × $3.00/1M
     = $0.00015 + $0.000402
     = $0.000552
```

**관리자 API**로 조회 가능:
```
GET /admin/usage?period=daily&key=YOUR_KEY
→ {total_requests: 15, total_cost_usd: 0.0082, avg_response_time_ms: 7500}
```

---

### 3.7 브리핑 생성 (`briefing/generator.py`)

#### 6종 브리핑 타입

| 타입 | 기간 | 기준 |
|------|------|------|
| daily | 오늘 00:00 ~ 현재 | 오늘 자정부터 |
| yesterday | 어제 00:00 ~ 오늘 00:00 | 어제 하루 |
| weekly | 이번주 월요일 00:00 ~ 현재 | 월요일 기준 |
| last_week | 지난주 월요일 ~ 이번주 월요일 | 정확히 7일 |
| monthly | 이번달 1일 00:00 ~ 현재 | 월초 기준 |
| last_month | 지난달 1일 ~ 이번달 1일 | 전월 전체 |

#### 브리핑 생성 흐름

```python
async def generate(self, briefing_type="daily"):
    # 1. 날짜 범위 계산 (캘린더 기반)
    start, end = self._get_date_range(briefing_type)

    # 2. 데이터 수집 (벡터 검색 아님, 날짜 필터링)
    data = self._collect_briefing_data(briefing_type, start, end)
    #   → recent_files:     Dropbox 파일 (created_date 범위)
    #   → received_emails:  외부 수신 메일
    #   → sent_emails:      DnS 직원 발신 메일

    # 3. LLM 프롬프트 구성
    prompt = self._build_prompt(briefing_type, data, ...)
    # "== 변동된 파일 (5건) =="
    # "1. [계약서.pdf] (CAS) - 2026-02-21"
    # "== 받은 메일 (3건) =="
    # ...

    # 4. LLM 호출 (브리핑 전용 시스템 프롬프트)
    content, usage = await self.generator._call_with_fallback(
        prompt,
        system_instruction="업무 브리핑을 작성하는 AI...",
        max_output_tokens=2048,
    )

    # 5. 출처 추가 + DB 저장 + 비용 로깅
    self._save_briefing(briefing_type, content)
    return content
```

**보낸/받은 메일 분류**: DnS 직원 이메일 주소로 구분합니다.
```python
_DNS_STAFF_EMAILS = {"theking57@naver.com", "ruthkim2015@naver.com"}

for email in all_emails:
    if email["email_from"].lower() in _DNS_STAFF_EMAILS:
        sent_emails.append(email)      # 직원이 보낸 메일
    else:
        received_emails.append(email)  # 외부에서 받은 메일
```

---

## 4. 데이터 흐름 다이어그램

### 4.1 사용자가 "A사 계약서 납품 기한?"을 물었을 때

```
카카오톡 → POST /skill/query
                │
                ▼
        skill_handler.py
        ├─ 브리핑 키워드? NO
        ├─ callbackUrl 있음? YES
        │   ├─ 즉시 응답: "🔍 검색 중..." (0.1초)
        │   └─ 백그라운드 태스크 시작
        │
        ▼ (백그라운드)
    callback.py
        │
        ▼
    chain.py — RAGChain.run()
        │
        ├─① retriever.search_and_prepare()
        │   ├─ embedder.embed("A사 계약서 납품 기한?") → [0.12, -0.34, ...]
        │   ├─ Zilliz.search(vector, limit=5)
        │   │   → Hit 1: "A사 납품계약서 ... 기한 3/15" (유사도 0.85)
        │   │   → Hit 2: "A사 발주서 ... 수량 100대"    (유사도 0.72)
        │   │   → Hit 3: "B사 계약서 ..."               (유사도 0.45)
        │   └─ format_context() → "[문서1] [파일: A사_납품계약.pdf]\n..."
        │
        ├─② generator.generate(query, context, sources)
        │   ├─ 프롬프트 조합: 시스템 + 컨텍스트 + 질문
        │   ├─ Gemini 3 Flash API 호출
        │   └─ → "A사 납품 기한은 2025년 3월 15일입니다.\n출처: A사_납품계약.pdf"
        │
        ├─③ _truncate(answer, 1000)
        │
        └─④ chat_logger.log_chat() ← fire-and-forget
                │
                ▼
        POST callbackUrl (1회용)
                │
                ▼
        카카오톡에 최종 답변 표시 (8~15초 후)
```

### 4.2 30분마다 데이터 동기화가 실행될 때

```
GitHub Actions (cron: */30 * * * *)
        │
        ▼
    scripts/full_sync.py
        │
        ├─ DropboxSync().sync()
        │   ├─ sync_state에서 cursor 로드
        │   ├─ Dropbox API: files_list_folder_continue(cursor)
        │   │   → FileMetadata(계약서_v2.pdf)    ← 수정됨
        │   │   → DeletedMetadata(old_memo.txt)  ← 삭제됨
        │   │
        │   ├─ 수정 파일 처리:
        │   │   ├─ 다운로드 → /tmp/계약서_v2.pdf
        │   │   ├─ text_extractor.extract_text() → "계약 내용..."
        │   │   ├─ chunker.split() → [Chunk(0), Chunk(1), Chunk(2)]
        │   │   └─ indexer.index_document()
        │   │       ├─ 기존 rows 삭제 (source_id="dropbox:id:xxx")
        │   │       ├─ embed_batch(3개 청크) → 3개 벡터
        │   │       └─ Zilliz insert(3행)
        │   │
        │   ├─ 삭제 파일 처리:
        │   │   └─ indexer.delete_document("dropbox:id:yyy")
        │   │
        │   └─ 새 cursor를 sync_state에 저장
        │
        └─ NaverMailSync().sync()
            ├─ sync_state에서 last_sync_date 로드
            ├─ 14개 폴더 순회
            │   ├─ IMAP SINCE 쿼리
            │   └─ 각 메일: 본문 추출 + 첨부파일 추출 → 청킹 → 인덱싱
            └─ last_sync_date 업데이트
```

### 4.3 평일 19시 자동 브리핑이 생성될 때

```
GitHub Actions (cron: 0 10 * * 1-5, UTC)
        │
        ▼
    scripts/run_briefing.py
        ├─ 요일 판단: 금요일? → weekly, 그 외 → daily
        │
        ▼
    BriefingGenerator().generate("daily")
        ├─ _get_date_range("daily") → (2026-02-21T00:00, 2026-02-21T19:00)
        ├─ _collect_briefing_data()
        │   ├─ Zilliz query: created_date 범위, source_type="dropbox"
        │   │   → 파일 5건
        │   ├─ Zilliz query: created_date 범위, source_type="email"
        │   │   → 메일 8건 (받은 5건, 보낸 3건)
        │   └─ 직원 이메일로 보낸/받은 분류
        │
        ├─ _build_prompt() → LLM 프롬프트 조합
        ├─ Gemini API 호출 → 브리핑 텍스트 생성
        ├─ _format_sources() → "[출처]" 섹션 추가
        └─ _save_briefing() → briefings 컬렉션에 저장
                │
                ▼
        (사용자가 "오늘 브리핑" 입력 시 저장된 브리핑 반환)
```

---

## 5. 설계 패턴 정리

### 5.1 싱글턴 (Singleton)

**한 번만 만들어서 공유하는 객체**.
DB 커넥션, API 클라이언트처럼 생성 비용이 높은 자원에 사용합니다.

```python
# 모듈 변수 방식 (zilliz_client.py)
_client = None
def get_client():
    global _client
    if _client is None:
        _client = MilvusClient(...)
    return _client

# 클래스 변수 방식 (embedder.py)
class Embedder:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 5.2 디스패치 테이블 (Dispatch Table)

**확장자/타입에 따라 적절한 함수를 선택**.
if-elif 체인보다 깔끔하고 확장이 쉽습니다.

```python
# text_extractor.py
_EXTRACTOR_MAP = {".pdf": _extract_pdf, ".docx": _extract_docx, ...}
def extract_text(path):
    return _EXTRACTOR_MAP[path.suffix](path)

# briefing/generator.py
_PROMPTS = {"daily": _DAILY_PROMPT, "weekly": _WEEKLY_PROMPT, ...}
```

### 5.3 폴백 (Fallback)

**주 경로 실패 시 대안 경로 시도**.

```
Gemini 3 Flash 실패 → Gemini 2.5 Flash 시도 → 에러 메시지 반환
PDF 텍스트 추출 실패 → Apple Vision OCR → PaddleOCR
HWP gethwp 실패 → pyhwp/hwp5txt
배치 insert 실패 → 개별 insert 재시도
```

### 5.4 Fire-and-Forget

**결과를 기다리지 않는 비동기 작업**.
응답 속도에 영향을 주면 안 되는 부가 작업(로깅 등)에 사용합니다.

```python
# 사용자 응답을 먼저 보내고, 로깅은 백그라운드에서
asyncio.create_task(chat_logger.log_chat(...))  # 결과 안 기다림
return answer  # 즉시 반환
```

### 5.5 커서 기반 증분 처리

**마지막으로 처리한 지점을 기억**해서 변경분만 처리합니다.

```
첫 실행: 전체 스캔 → 커서 "abc123" 저장
2번째:   커서 "abc123" 이후 변경분만 → 커서 "def456" 저장
3번째:   커서 "def456" 이후 변경분만 → ...
```

---

## 6. 코드 규모 요약

| 레이어 | 파일 수 | 코드 줄 수 | 핵심 역할 |
|-------|---------|-----------|----------|
| config | 1 | 55 | 환경 변수 |
| db | 1 | 207 | DB 연결 + 스키마 |
| ingestion | 5 | 1,908 | 데이터 수집 |
| rag | 4 | 901 | 검색 + 답변 생성 |
| server | 5 | 674 | API + 로깅 |
| briefing | 2 | 748 | 브리핑 생성 |
| scripts | 7 | 1,115 | 유틸리티 |
| **합계** | **25** | **~5,600** | |

---

## 7. 인프라 비용

| 서비스 | 용도 | 월 비용 |
|-------|------|--------|
| Zilliz Cloud | 벡터 DB (5GB 무료) | ₩0 |
| Render | 웹 서버 (무료 티어) | ₩0 |
| GitHub Actions | 스케줄링 (무료) | ₩0 |
| Gemini API | LLM + 임베딩 | ~₩7,000 |
| Dropbox | 파일 저장소 (기존) | ₩0 |
| **합계** | | **~₩7,000/월** |
