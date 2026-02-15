# Dropbox + 네이버 메일 기반 RAG 카카오톡 챗봇 개발 계획

## 프로젝트 개요

Dropbox 업무 파일과 네이버 메일을 데이터 소스로 하는 RAG(Retrieval-Augmented Generation) 시스템을 구축하고, 카카오톡 채널 챗봇을 통해 3명의 팀원이 자연어로 업무 문서를 검색/질의할 수 있는 시스템.

### 핵심 목표
- **비용**: 월 ~₩7,000 (LLM API 외 모든 서비스 무료 티어 활용)
- **사용자**: 팀원 3명 (비공개 카카오톡 채널)
- **기능**: RAG 질의응답 + 자동 브리핑 (일간/주간/월간)

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        데이터 소스                                │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │   Dropbox     │    │   네이버 메일      │                       │
│  │  (API/SDK)    │    │   (IMAP)         │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │                     │                                  │
│         ▼                     ▼                                  │
│  ┌─────────────────────────────────────┐                        │
│  │     데이터 수집 & 전처리 파이프라인       │                        │
│  │  - 텍스트 추출 (PyMuPDF, python-docx)  │                       │
│  │  - 청킹 (500~1000 토큰)               │                       │
│  │  - 임베딩 (Gemini text-embedding-004) │                       │
│  └──────────────┬──────────────────────┘                        │
│                 ▼                                                │
│  ┌─────────────────────────────────────┐                        │
│  │   Supabase (PostgreSQL + pgvector)   │                        │
│  │   - 벡터 저장 & 유사도 검색             │                        │
│  │   - 메타데이터 필터링                   │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ▼                                                │
│  ┌─────────────────────────────────────┐                        │
│  │      스킬 서버 (FastAPI on Render)    │                        │
│  │  - /skill/query  → RAG 질의응답       │                        │
│  │  - /skill/briefing → 브리핑 요청       │                        │
│  │  - Callback API 지원 (5초 타임아웃 해결)│                        │
│  └──────────────┬──────────────────────┘                        │
│                 ▼                                                │
│  ┌─────────────────────────────────────┐                        │
│  │   Gemini 3 Flash (LLM)              │                        │
│  │   - $0.50/$3.00 per 1M tok (I/O)   │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ▼                                                │
│  ┌─────────────────────────────────────┐                        │
│  │   카카오톡 채널 챗봇                    │                        │
│  │  - 비공개 채널 (검색 OFF, URL 공유)     │                        │
│  │  - 카카오 i 오픈빌더 + 스킬 연동         │                        │
│  │  - AI 챗봇 전환 (콜백 활성화)           │                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │   GitHub Actions (스케줄링)            │                        │
│  │  - 데이터 동기화 (30분 주기)            │                        │
│  │  - 브리핑 생성 (매일 09:00 KST)        │                        │
│  └─────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 구성 요소 | 기술 | 비용 |
|-----------|------|------|
| 벡터 DB | Supabase Free (PostgreSQL + pgvector) | ₩0 |
| 임베딩 | Gemini `text-embedding-004` API (무료 티어) | ₩0 |
| LLM | Google Gemini 3 Flash (유료, 폴백: 2.5 Flash) | ~₩7,000/월 |
| 스킬 서버 | FastAPI on Render 무료 웹서비스 | ₩0 |
| 챗봇 | 카카오톡 채널 + 카카오 i 오픈빌더 | ₩0 |
| 스케줄링 | GitHub Actions cron | ₩0 |
| 데이터 수집 | Dropbox SDK + imap_tools | ₩0 |
| 텍스트 추출 | PyMuPDF, python-docx, openpyxl, python-pptx, BeautifulSoup | ₩0 |
| **합계** | | **~₩7,000/월** |

---

## 프로젝트 구조

```
project-root/
├── README.md
├── requirements.txt
├── .env.example                  # 환경 변수 템플릿
├── .github/
│   └── workflows/
│       ├── sync_data.yml         # 데이터 동기화 (30분 주기)
│       └── daily_briefing.yml    # 브리핑 생성 (매일 09:00 KST)
├── src/
│   ├── __init__.py
│   ├── config.py                 # 환경 변수 및 설정
│   ├── server/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 앱 진입점
│   │   ├── skill_handler.py      # 카카오 스킬 요청/응답 처리
│   │   └── callback.py           # 콜백 API 비동기 처리
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py           # 임베딩 모델 래퍼
│   │   ├── retriever.py          # Supabase 벡터 검색
│   │   ├── generator.py          # Gemini LLM 호출
│   │   └── chain.py              # RAG 체인 (검색 → 생성)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── dropbox_sync.py       # Dropbox 변경 감지 및 동기화
│   │   ├── naver_mail_sync.py    # 네이버 메일 IMAP 동기화
│   │   ├── text_extractor.py     # 파일 형식별 텍스트 추출
│   │   ├── chunker.py            # 텍스트 청킹
│   │   └── indexer.py            # Supabase 벡터 인덱싱
│   ├── briefing/
│   │   ├── __init__.py
│   │   ├── generator.py          # 브리핑 생성 (일간/주간/월간)
│   │   └── sender.py             # 카카오톡 채널 메시지 발송
│   └── db/
│       ├── __init__.py
│       ├── supabase_client.py    # Supabase 클라이언트
│       └── schema.sql            # DB 스키마 (pgvector 테이블)
├── scripts/
│   ├── init_db.py                # 초기 DB 셋업
│   ├── full_sync.py              # 전체 데이터 최초 동기화
│   └── run_briefing.py           # 브리핑 수동 실행
└── tests/
    ├── test_embedder.py
    ├── test_retriever.py
    ├── test_dropbox_sync.py
    └── test_skill_handler.py
```

---

## 개발 단계

### Phase 1: 인프라 셋업 (1~2일)

#### 1-1. Supabase 프로젝트 생성 및 DB 스키마

Supabase Free 티어에서 프로젝트를 생성하고 pgvector 확장을 활성화한다.

**DB 스키마** (`src/db/schema.sql`):

```sql
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 통합 문서 테이블 (Dropbox + 메일 단일 인덱스)
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,           -- 'dropbox' 또는 'email'
    source_id TEXT NOT NULL,              -- Dropbox file_id 또는 email message_id
    content TEXT NOT NULL,               -- 청크 텍스트
    embedding VECTOR(768),              -- Gemini text-embedding-004 차원
    
    -- 공통 메타데이터
    created_date TIMESTAMPTZ,
    updated_date TIMESTAMPTZ DEFAULT NOW(),
    
    -- Dropbox 메타데이터
    filename TEXT,
    folder_path TEXT,
    file_type TEXT,                      -- 'pdf', 'docx', 'xlsx', 'pptx'
    
    -- 이메일 메타데이터
    email_from TEXT,
    email_to TEXT,
    email_subject TEXT,
    email_date TIMESTAMPTZ,
    
    -- 청크 정보
    chunk_index INTEGER DEFAULT 0,

    UNIQUE(source_id, chunk_index)        -- 복합 유니크 (하나의 문서에서 여러 청크)
);

-- 벡터 유사도 검색 인덱스 (IVFFlat)
CREATE INDEX ON documents 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- 메타데이터 필터링 인덱스
CREATE INDEX idx_source_type ON documents(source_type);
CREATE INDEX idx_created_date ON documents(created_date);
CREATE INDEX idx_source_id ON documents(source_id);

-- 동기화 상태 테이블
CREATE TABLE sync_state (
    id SERIAL PRIMARY KEY,
    sync_type TEXT NOT NULL UNIQUE,      -- 'dropbox' 또는 'email'
    last_cursor TEXT,                    -- Dropbox cursor
    last_sync_time TIMESTAMPTZ,         -- 마지막 동기화 시점
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 브리핑 히스토리
CREATE TABLE briefings (
    id BIGSERIAL PRIMARY KEY,
    briefing_type TEXT NOT NULL,         -- 'daily', 'weekly', 'monthly'
    content TEXT NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    sent BOOLEAN DEFAULT FALSE
);

-- 벡터 유사도 검색 함수
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_count INT DEFAULT 5,
    filter_source_type TEXT DEFAULT NULL,
    filter_after_date TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    source_type TEXT,
    content TEXT,
    filename TEXT,
    email_subject TEXT,
    email_from TEXT,
    created_date TIMESTAMPTZ,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.source_type,
        d.content,
        d.filename,
        d.email_subject,
        d.email_from,
        d.created_date,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 
        (filter_source_type IS NULL OR d.source_type = filter_source_type)
        AND (filter_after_date IS NULL OR d.created_date >= filter_after_date)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

#### 1-2. 환경 변수 설정

**`.env.example`**:

```env
# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIs...

# Dropbox
DROPBOX_ACCESS_TOKEN=sl.xxxxx
DROPBOX_FOLDER_PATH=/업무  # 동기화 대상 루트 폴더

# 네이버 메일
NAVER_EMAIL=your_email@naver.com
NAVER_PASSWORD=your_app_password  # 2단계 인증 시 앱 비밀번호
NAVER_IMAP_SERVER=imap.naver.com

# Google Gemini
GEMINI_API_KEY=AIzaSy...

# 카카오톡 채널 (브리핑 발송용)
KAKAO_ADMIN_KEY=xxxxx
KAKAO_CHANNEL_ID=xxxxx
```

#### 1-3. 카카오톡 채널 및 챗봇 셋업

1. **카카오비즈니스** (business.kakao.com) 가입 및 채널 생성
2. 채널 설정: **홈 공개 ON + 검색 허용 OFF** (URL 아는 사람만 접근)
3. **챗봇 관리자센터** (i.kakao.com) → 봇 만들기 → 카카오톡 챗봇
4. 설정 → AI 챗봇 관리 → 콜백 기능 신청 (영업일 1~2일 소요)
5. 채널 URL을 팀원 3명에게만 공유하여 친구 추가

---

### Phase 2: 데이터 수집 & 인덱싱 파이프라인 (2~3일)

#### 2-1. 텍스트 추출기

**`src/ingestion/text_extractor.py`**:

지원 파일 형식 및 라이브러리:

| 파일 형식 | 라이브러리 | 비고 |
|-----------|-----------|------|
| PDF | `PyMuPDF` (fitz) | OCR 불필요한 텍스트 PDF |
| DOCX | `python-docx` | 테이블 포함 |
| XLSX | `openpyxl` | 시트별 추출 |
| PPTX | `python-pptx` | 슬라이드별 추출 |
| **HWP** | `gethwp` → `pyhwp`(hwp5txt) 폴백 | 한글 워드프로세서 바이너리(OLE) |
| **HWPX** | `gethwp` → ZIP/XML 직접 파싱 폴백 | 한글 워드프로세서 XML 형식 |
| **CELL** | `openpyxl` (확장자 변환) | 한셀 스프레드시트 (OpenXML 호환) |
| **ZIP** | `zipfile` (내장) | 압축 해제 후 내부 파일 재귀 처리 |
| 이메일 HTML | `BeautifulSoup` | 시그니처/광고 필터링 |
| TXT/CSV | 내장 | 직접 읽기 |

구현 요구사항:
- 각 파일 형식별 추출 함수를 dispatch 패턴으로 연결
- 추출 실패 시 에러 로깅하고 건너뛰기 (전체 동기화 중단 방지)
- 이메일 HTML에서 시그니처, 광고, 면책조항 등 제거
- HWP/HWPX는 다단계 폴백 전략 사용 (gethwp → pyhwp/hwp5txt → LibreOffice headless)
- **ZIP 처리 전략**:
  - `zipfile` (Python 내장)로 압축 해제 → 임시 디렉토리에 풀기
  - 내부 파일을 지원 확장자 기준으로 필터링 후 재귀적으로 텍스트 추출
  - 중첩 ZIP (ZIP 안의 ZIP)은 최대 2단계까지만 처리
  - 암호화된 ZIP은 스킵 및 로깅
  - 해제 후 총 파일 크기 50MB 초과 시 스킵 (zip bomb 방지)
  - 각 내부 파일의 메타데이터에 원본 ZIP 경로 + 내부 파일 경로 기록
  - `source_id` 형식: `{zip_file_id}:{internal_path}` (청크 단위 삭제/업데이트 가능)

#### 2-2. 텍스트 청킹

**`src/ingestion/chunker.py`**:

- `RecursiveCharacterTextSplitter` 스타일의 청킹
- **chunk_size**: 500자 (한국어 기준)
- **chunk_overlap**: 50자
- 각 청크에 원본 문서의 메타데이터 유지

#### 2-3. 임베딩

**`src/rag/embedder.py`**:

```python
from google import genai
from src.config import settings

class Embedder:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = "text-embedding-004"

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model, contents=text
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
```

- 모델: Gemini `text-embedding-004` (768차원)
- API 기반 → 로컬 모델 다운로드/GPU 불필요
- Render 무료 티어 (512MB RAM)에서 구동 가능
- 무료 티어: 분당 1,500회 → 3명 사용 + 동기화에 충분

#### 2-4. Dropbox 동기화

**`src/ingestion/dropbox_sync.py`**:

동기화 전략: **Cursor 기반 증분 동기화**

```
1. 최초 실행: files_list_folder()로 전체 목록 가져오기 → cursor 저장
2. 이후 실행: files_list_folder_continue(cursor)로 변경분만 감지
3. 변경 유형 처리:
   - 파일 추가/수정 → 텍스트 추출 → 청킹 → 임베딩 → Supabase upsert
   - 파일 삭제 → Supabase에서 해당 source_id 삭제
4. cursor를 sync_state 테이블에 저장
```

구현 요구사항:
- `dropbox` Python SDK 사용
- 지원 확장자 필터링: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.hwp`, `.hwpx`, `.cell`, `.txt`, `.csv`, `.zip`
- 대용량 파일(>10MB) 스킵 및 로깅
- 동기화 주기: 30분 (GitHub Actions cron)

#### 2-5. 네이버 메일 동기화

**`src/ingestion/naver_mail_sync.py`**:

동기화 전략: **SINCE 쿼리 기반 증분 동기화**

```
1. sync_state에서 마지막 동기화 시점 조회
2. IMAP SINCE 쿼리로 신규 메일 조회 (수신함 + 보낸메일함)
3. 각 메일 처리:
   - 본문 HTML → 텍스트 변환 (시그니처/광고 제거)
   - 첨부파일 추출 → 텍스트 변환
   - 청킹 → 임베딩 → Supabase upsert
4. 마지막 동기화 시점 업데이트
```

구현 요구사항:
- `imap_tools` 라이브러리 사용 (imaplib보다 편리)
- IMAP 서버: `imap.naver.com:993` (SSL)
- 수신함(INBOX) + 보낸메일함(Sent) 동시 동기화
- 동기화 주기: 30분 (Dropbox와 동일 스케줄)

#### 2-6. Supabase 인덱싱

**`src/ingestion/indexer.py`**:

```
파일 수정 시:
1. supabase.table("documents").delete().eq("source_id", source_id).execute()
2. 새 청크들을 upsert
→ source_id 기준으로 기존 청크 모두 삭제 후 재삽입
```

구현 요구사항:
- `supabase-py` 라이브러리 사용
- 배치 upsert (한 번에 최대 100개 청크)
- 에러 발생 시 개별 청크 단위로 재시도

---

### Phase 3: RAG 파이프라인 (1~2일)

#### 3-1. 벡터 검색

**`src/rag/retriever.py`**:

```python
async def search(
    query: str,
    source_type: str | None = None,  # 'dropbox', 'email', None(전체)
    after_date: str | None = None,
    top_k: int = 5
) -> list[dict]:
    query_vector = embedder.embed(query)
    results = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_count": top_k,
        "filter_source_type": source_type,
        "filter_after_date": after_date
    }).execute()
    return results.data
```

#### 3-2. LLM 응답 생성

**`src/rag/generator.py`**:

- Google Gemini 3 Flash API 사용 (`gemini-3-flash-preview` 모델)
- GA 전환 시 `gemini-3-flash`로 변경, 폴백: `gemini-2.5-flash`
- `google-genai` 라이브러리 (구 `google-generativeai`는 2025.11.30 EOL)
- 유료 API: 입력 $0.50/1M, 출력 $3.00/1M 토큰 (월 ~$5 예상)
- 시스템 프롬프트에 한국어 응답 + 출처 표기 지시

**시스템 프롬프트**:

```
당신은 팀의 업무 문서와 이메일을 기반으로 질문에 답변하는 AI 어시스턴트입니다.

규칙:
1. 제공된 문서 컨텍스트만을 기반으로 답변하세요.
2. 문서에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 답변 끝에 출처(파일명 또는 이메일 제목)를 표기하세요.
4. 한국어로 답변하세요.
5. 간결하되 핵심 정보를 빠뜨리지 마세요.
```

#### 3-3. RAG 체인

**`src/rag/chain.py`**:

```
1. 사용자 질문 수신
2. 질문 임베딩
3. Supabase 벡터 검색 (top 5)
4. 검색 결과를 컨텍스트로 구성
5. Gemini 3 Flash에 프롬프트 전달 (시스템 프롬프트 + 컨텍스트 + 질문)
6. 응답 생성 + 출처 정보 포매팅
7. 응답 반환 (카카오톡 1000자 제한 준수)
```

---

### Phase 4: 스킬 서버 (카카오톡 연동) (2~3일)

#### 4-1. FastAPI 스킬 서버

**`src/server/main.py`**:

FastAPI 기반 서버. 카카오 오픈빌더의 스킬 요청을 받아 RAG 응답을 반환한다.

엔드포인트:

| 엔드포인트 | 용도 |
|-----------|------|
| `POST /skill/query` | RAG 질의응답 (폴백 블록 연결) |
| `POST /skill/briefing` | 브리핑 요청 |
| `GET /health` | 헬스체크 (Render 무료 티어 슬립 방지) |

#### 4-2. 카카오 스킬 요청/응답 처리

**`src/server/skill_handler.py`**:

카카오 오픈빌더 스킬 요청 형식:

```json
{
  "intent": { "name": "..." },
  "userRequest": {
    "utterance": "A사 계약서 납품 기한 알려줘",
    "callbackUrl": "https://bot-api.kakao.com/callback/..."  // AI 챗봇 전환 시
  },
  "bot": { "id": "..." },
  "action": { "params": {} }
}
```

응답 형식:

```json
{
  "version": "2.0",
  "template": {
    "outputs": [
      {
        "simpleText": {
          "text": "답변 내용 (최대 1000자)"
        }
      }
    ]
  }
}
```

#### 4-3. 콜백 API 처리 (5초 타임아웃 해결)

**`src/server/callback.py`**:

카카오 스킬 서버는 5초 내 응답 필수. RAG 처리는 5초 이상 걸릴 수 있으므로 콜백 사용.

```
1. 스킬 요청 수신
2. callbackUrl 존재 확인
3. 즉시 응답: {"useCallback": true, "template": {"outputs": [{"simpleText": {"text": "🔍 검색 중..."}}]}}
4. 백그라운드 태스크로 RAG 처리
5. 처리 완료 후 callbackUrl로 POST (최종 응답)
```

콜백 제약사항:
- callbackUrl 유효시간: **1분**
- 사용 횟수: **1회**
- 봇테스트에서 테스트 불가 → 반드시 배포 후 실제 카카오톡에서 테스트

구현:

```python
@app.post("/skill/query")
async def skill_query(request: Request):
    body = await request.json()
    utterance = body["userRequest"]["utterance"]
    callback_url = body["userRequest"].get("callbackUrl")

    if callback_url:
        # 콜백 모드: 비동기 처리
        asyncio.create_task(process_and_callback(utterance, callback_url))
        return {
            "version": "2.0",
            "useCallback": True,
            "template": {
                "outputs": [{"simpleText": {"text": "🔍 문서를 검색하고 있습니다..."}}]
            }
        }
    else:
        # 콜백 미지원: 5초 내 빠른 응답 시도
        answer = await quick_rag(utterance)
        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer[:1000]}}]
            }
        }


async def process_and_callback(utterance: str, callback_url: str):
    answer = await rag_chain.run(utterance)
    formatted = format_answer_with_source(answer)

    async with httpx.AsyncClient() as client:
        await client.post(callback_url, json={
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": formatted[:1000]}}]
            }
        })
```

#### 4-4. 카카오 오픈빌더 설정

1. **스킬 등록**: 챗봇 관리자센터 → 스킬 → URL 입력 (`https://your-app.onrender.com/skill/query`)
2. **폴백 블록에 스킬 연결**: 모든 미분류 발화가 RAG 스킬로 라우팅되도록 설정
3. **별도 블록 (선택)**: "오늘 브리핑", "이번 주 브리핑" 등 패턴 발화 블록 추가
4. **콜백 설정**: AI 챗봇 전환 승인 후, 해당 블록에서 콜백 옵션 ON
5. **배포**: 전체 배포 실행

#### 4-5. Render 배포

- Render 무료 웹서비스 (750시간/월)
- `render.yaml` 또는 대시보드에서 설정
- 환경 변수: `.env`의 모든 키를 Render 대시보드에 등록
- 주의: 무료 티어는 15분 비활성 시 슬립 → `/health` 엔드포인트를 외부에서 주기적 ping

---

### Phase 5: 자동 브리핑 (1~2일)

#### 5-1. 브리핑 생성기

**`src/briefing/generator.py`**:

| 브리핑 타입 | 스케줄 | 대상 기간 |
|------------|--------|----------|
| 일간 | 매일 09:00 KST | 어제 ~ 오늘 |
| 주간 | 매주 월요일 09:00 KST | 지난 주 |
| 월간 | 매월 1일 09:00 KST | 지난 달 |

브리핑 생성 프로세스:

```
1. Supabase에서 기간별 문서 검색 (created_date 필터)
2. 벡터 검색으로 예정 업무/마감 관련 문서 추출
   - 검색 키워드: "일정", "마감", "deadline", "회의", "미팅", "예정"
3. Gemini 3 Flash 프롬프트로 요약 생성:
   - [기간] 업무 요약 (주요 활동)
   - [예정] 할 일 / 마감 임박 항목
   - [중요] 주의 사항
4. 브리핑을 briefings 테이블에 저장
5. 카카오톡 채널 메시지로 3명에게 발송
```

브리핑 프롬프트 예시:

```
다음은 {시작일}~{종료일} 기간의 업무 문서/이메일 목록입니다.

{문서 목록}

위 내용을 바탕으로 다음 형식의 업무 브리핑을 작성하세요:

📋 {브리핑 타입} 브리핑 ({날짜})

[지난 기간 업무 요약]
• 주요 활동 3~5개

[향후 할 일]  
⚠️ 마감 임박 항목
• 예정된 업무 목록

[기타 참고사항]
• 중요 공지나 변경 사항
```

#### 5-2. 카카오톡 채널 메시지 발송

**`src/briefing/sender.py`**:

카카오톡 채널 메시지 API를 통해 채널 친구(3명)에게 브리핑 발송.

> 참고: 카카오 비즈메시지 API 또는 채널 1:1 채팅 API 사용. 
> 무료 발송 한도 내에서 운영 (일 1,000건 이내).

#### 5-3. GitHub Actions 스케줄

**`.github/workflows/daily_briefing.yml`**:

```yaml
name: Daily Briefing
on:
  schedule:
    - cron: '0 0 * * *'    # UTC 00:00 = KST 09:00 (일간)
    - cron: '0 0 * * 1'    # UTC 00:00 = KST 09:00 월요일 (주간)
  workflow_dispatch:         # 수동 실행 가능

jobs:
  briefing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python scripts/run_briefing.py
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          # ... 기타 환경 변수
```

**`.github/workflows/sync_data.yml`**:

```yaml
name: Data Sync
on:
  schedule:
    - cron: '*/30 * * * *'   # 30분마다
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python scripts/full_sync.py
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          DROPBOX_ACCESS_TOKEN: ${{ secrets.DROPBOX_ACCESS_TOKEN }}
          NAVER_EMAIL: ${{ secrets.NAVER_EMAIL }}
          NAVER_PASSWORD: ${{ secrets.NAVER_PASSWORD }}
```

---

## 주요 라이브러리 (requirements.txt)

```
# 웹 서버
fastapi==0.129.*
uvicorn==0.40.*
httpx==0.28.*

# Supabase
supabase==2.24.*

# LLM + 임베딩 (google-genai가 LLM과 임베딩 모두 처리)
google-genai==1.*

# 데이터 수집
dropbox==12.*
imap-tools==1.*

# 텍스트 추출
PyMuPDF==1.27.*
python-docx==1.*
openpyxl==3.*
python-pptx==1.*
beautifulsoup4==4.*
gethwp==1.*                # HWP/HWPX 텍스트 추출
pyhwp==0.1b15              # HWP 폴백 (hwp5txt CLI 포함)
lxml                       # HWPX XML 파싱용

# 유틸리티
python-dotenv==1.*
pydantic==2.*
pydantic-settings          # BaseSettings 환경변수 관리
```

---

## Supabase Free 티어 제약 및 대응

| 제약 | 한도 | 대응 |
|------|------|------|
| DB 용량 | 500MB | 임베딩(768차원 float32) + 텍스트로 약 5만~10만 청크 수용 가능. 충분 |
| 파일 스토리지 | 1GB | 사용하지 않음 (원본은 Dropbox에 보관) |
| 비활성 일시정지 | 1주 | 스킬 서버에서 주기적 DB 쿼리로 활성 유지 |
| 활성 프로젝트 | 2개 | 1개만 사용 |
| 인덱스 제한 | 3개 | 단일 통합 테이블 + source_type 필터로 해결 |

---

## 카카오톡 채널 비공개 운영 설정

| 설정 항목 | 값 | 효과 |
|----------|-----|------|
| 홈 공개 | ON | 채널 URL로 접근 가능 |
| 검색 허용 | OFF | 카카오톡 검색에 노출 안 됨 |

채널 URL (`https://pf.kakao.com/_xxxxx`)을 팀원 3명에게만 공유.
외부 사용자가 검색으로 발견할 수 없어 사실상 비공개 운영.

---

## 사용자 경험 예시

```
[카카오톡 채널 채팅]

사용자: A사 계약서 납품 기한 알려줘
봇:     🔍 문서를 검색하고 있습니다...
봇:     A사 납품계약서에 따르면 납품 기한은 2025년 3월 15일입니다.
        📎 출처: Dropbox/계약서/A사_납품계약_2025.pdf

사용자: 지난주에 김과장이 보낸 메일 중에 회의 관련 내용 있어?
봇:     🔍 문서를 검색하고 있습니다...
봇:     김과장님이 2/10에 보낸 "2월 정기회의 안건" 메일에서:
        - 2월 정기회의: 2/20(목) 14:00 본사 3층
        - 안건: Q1 실적 리뷰, 신규 프로젝트 킥오프
        📎 출처: 이메일 - "2월 정기회의 안건" (2025-02-10)

사용자: 오늘 브리핑
봇:     📋 오늘의 업무 브리핑 (2/17 월)

        [어제 업무 요약]
        • A사 계약서 수정본 Dropbox 업로드
        • B사 미팅 후기 메일 수신

        [오늘 할 일]
        ⚠️ A사 계약서 최종본 회신 (마감: 오늘)
        • 14:00 B사 화상회의

        [참고사항]
        • 내일 정기회의 안건 준비 필요
```

---

## 구현 우선순위 및 타임라인

| 순서 | 단계 | 예상 소요 | 의존성 |
|------|------|----------|--------|
| 1 | 카카오 채널/챗봇 셋업 + AI 챗봇 신청 | 1일 (+ 승인 1~2일) | 없음 |
| 2 | Supabase 프로젝트 + DB 스키마 | 0.5일 | 없음 |
| 3 | 텍스트 추출 + 청킹 모듈 | 1일 | 없음 |
| 4 | 임베딩 + 인덱싱 모듈 | 0.5일 | Phase 2 |
| 5 | Dropbox 동기화 | 1일 | Phase 2, 4 |
| 6 | 네이버 메일 동기화 | 1일 | Phase 2, 4 |
| 7 | RAG 체인 (검색 + 생성) | 1일 | Phase 4 |
| 8 | FastAPI 스킬 서버 + 콜백 | 1일 | Phase 7 |
| 9 | 오픈빌더 스킬/블록 연결 + 배포 | 0.5일 | Phase 1, 8 |
| 10 | Render 배포 + 통합 테스트 | 0.5일 | Phase 8, 9 |
| 11 | 브리핑 생성 + 발송 | 1일 | Phase 7 |
| 12 | GitHub Actions 스케줄링 | 0.5일 | Phase 5, 6, 11 |
| **합계** | | **약 10일** | |

> Phase 1의 AI 챗봇 승인(1~2 영업일)을 기다리는 동안 Phase 2~7을 병행 개발하면 전체 일정을 단축할 수 있다.

---

## 주의사항 및 제약

1. **Gemini 3 Flash API 비용**: 입력 $0.50/1M, 출력 $3.00/1M 토큰. 월 ~$5 예상. Preview 모델이므로 GA 전환 시 가격 변동 가능. 폴백으로 Gemini 2.5 Flash 유지
2. **Render 무료 티어 슬립**: 15분 비활성 시 슬립. 첫 요청 시 30초~1분 콜드스타트 → UptimeRobot 등으로 5분 간격 ping 권장
3. **Supabase 비활성 일시정지**: 1주 미사용 시 자동 정지. 스킬 서버의 정기 쿼리로 방지
4. **카카오 콜백 테스트**: 봇테스트에서 불가. 반드시 배포 후 실제 카카오톡에서 테스트
5. **네이버 IMAP**: 2단계 인증 사용 시 앱 비밀번호 별도 발급 필요
6. **Dropbox 토큰**: 장기 토큰(offline access) 사용. 만료 시 refresh 로직 필요
7. **임베딩 API**: Gemini `text-embedding-004` 무료 티어 사용. 분당 1,500회 제한이므로 대량 인덱싱 시 rate limiting 필요
8. **Python 버전**: Python 3.12+ 사용. GitHub Actions 및 Render에서 동일 버전 유지
9. **Gemini SDK**: `google-generativeai`는 2025.11.30 EOL. 반드시 `google-genai` 사용. LLM과 임베딩 모두 동일 SDK로 처리
10. **Gemini 3 Flash Preview**: 아직 GA가 아님. 프로덕션 안정성을 위해 `gemini-2.5-flash`를 폴백 모델로 유지. GA 전환 시 모델 ID를 `gemini-3-flash`로 변경
