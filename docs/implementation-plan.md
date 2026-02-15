# DnS Trading RAG 카카오톡 챗봇 -- 구현 계획 (Agent Teams 병렬 실행)

## Context

CLAUDE.md에 상세 설계된 RAG 챗봇 프로젝트를 구현한다. 현재 코드는 전혀 없고 CLAUDE.md만 존재.

**아키텍처 변경**: 임베딩을 로컬 모델 대신 **Gemini text-embedding-004 API**로 전환.
이에 따라 `sentence-transformers`, `torch` 제거, DB 벡터 차원 1024→768 변경.

---

## 실행 전략: Agent Teams 병렬 구현

의존성 그래프에 따라 4단계로 나누어 병렬 실행한다.

```
[Step 1] 리드가 직접 실행 -- 스캐폴딩
    │
[Step 2] 에이전트 3개 병렬 ─┬─ Agent A: 텍스트 추출 + 청킹
    │                      ├─ Agent B: DB 클라이언트 + 임베더 + 인덱서
    │                      └─ Agent C: RAG 파이프라인 (retriever + generator + chain)
    │
[Step 3] 에이전트 3개 병렬 ─┬─ Agent D: Dropbox + Naver 동기화 + scripts
    │                      ├─ Agent E: FastAPI 스킬 서버 (카카오톡 연동)
    │                      └─ Agent F: 브리핑 시스템
    │
[Step 4] 에이전트 2개 병렬 ─┬─ Agent G: 배포 설정 (GitHub Actions, Procfile)
                           └─ Agent H: 테스트 코드
```

---

## Step 1: 스캐폴딩 (리드가 직접 실행)

모든 에이전트가 참조할 기반 파일을 먼저 생성한다.

**파일:**
- `.gitignore`
- `requirements.txt`
- `.env.example`
- `src/__init__.py`
- `src/config.py` -- pydantic BaseSettings, 앱 상수
- `src/db/__init__.py`
- `src/db/schema.sql` -- VECTOR(768), UNIQUE(source_id, chunk_index)
- `src/ingestion/__init__.py`
- `src/rag/__init__.py`
- `src/server/__init__.py`
- `src/briefing/__init__.py`

**requirements.txt:**
```
fastapi==0.129.*
uvicorn==0.40.*
httpx==0.28.*
supabase==2.24.*
google-genai==1.*
dropbox==12.*
imap-tools==1.*
PyMuPDF==1.27.*
python-docx==1.*
openpyxl==3.*
python-pptx==1.*
beautifulsoup4==4.*
gethwp==1.*
pyhwp==0.1b15
lxml
python-dotenv==1.*
pydantic==2.*
pydantic-settings
```

---

## Step 2: 핵심 모듈 병렬 구현 (3 에이전트)

### Agent A: 텍스트 추출 + 청킹
**담당 파일:**
- `src/ingestion/text_extractor.py`
- `src/ingestion/chunker.py`

**text_extractor.py 스펙:**
- `extract_text(path) -> str` -- 단일 파일
- `extract_files_from_archive(path) -> list[tuple[str, str]]` -- ZIP용
- PDF(fitz), DOCX(python-docx), XLSX(openpyxl), PPTX(python-pptx)
- HWP: gethwp → pyhwp(hwp5txt) 폴백
- HWPX: gethwp → ZIP/XML 직접 파싱 폴백
- CELL: openpyxl 처리
- ZIP: 재귀 2단계, 50MB 제한, 암호화 스킵
- HTML: BeautifulSoup, 시그니처/광고 제거
- TXT/CSV: 인코딩 자동 감지 (UTF-8 → EUC-KR → CP949)

**chunker.py 스펙:**
- chunk_size=500, chunk_overlap=50
- 분리자: `\n\n` → `\n` → `. ` → ` ` → 문자
- Chunk dataclass: text, chunk_index, metadata

### Agent B: DB 클라이언트 + 임베더 + 인덱서
**담당 파일:**
- `src/db/supabase_client.py`
- `src/rag/embedder.py`
- `src/ingestion/indexer.py`
- `scripts/init_db.py`

**supabase_client.py:** 싱글톤 Supabase 클라이언트
**embedder.py:** Gemini text-embedding-004, 768차원, embed() + embed_batch()
**indexer.py:** source_id 기준 삭제→배치 insert(100개), 실패 시 개별 재시도, DocumentMetadata dataclass
**init_db.py:** 테이블 존재 확인 스크립트

### Agent C: RAG 파이프라인
**담당 파일:**
- `src/rag/retriever.py`
- `src/rag/generator.py`
- `src/rag/chain.py`

**retriever.py:** Supabase match_documents RPC, 컨텍스트 포매팅, 출처 추출
**generator.py:** google-genai async API, gemini-3-flash-preview (폴백: 2.5-flash), 시스템 프롬프트 (한국어, 출처 표기), temperature=0.3
**chain.py:** run(query, top_k=5), quick_run(query, top_k=3), 1000자 제한

---

## Step 3: 응용 모듈 병렬 구현 (3 에이전트)

### Agent D: 데이터 동기화
**담당 파일:**
- `src/ingestion/dropbox_sync.py`
- `src/ingestion/naver_mail_sync.py`
- `scripts/full_sync.py`

**dropbox_sync.py:** Cursor 기반 증분, 파일 다운로드→추출→청킹→인덱싱, 삭제 처리, ZIP source_id `{file_id}:{internal_path}`
**naver_mail_sync.py:** imap_tools, INBOX+Sent, SINCE 쿼리, 본문+첨부파일, source_id `email:{uid}:body|att:{filename}`, mark_seen=False
**full_sync.py:** DropboxSync + NaverMailSync 순차 실행

### Agent E: FastAPI 스킬 서버
**담당 파일:**
- `src/server/main.py`
- `src/server/skill_handler.py`
- `src/server/callback.py`

**main.py:** FastAPI 앱, lifespan, /health, 라우터 등록
**skill_handler.py:** /skill/query (콜백/비콜백), /skill/briefing, 카카오 스킬 응답 형식
**callback.py:** process_and_callback (비동기 RAG → callbackUrl POST), process_briefing_and_callback

### Agent F: 브리핑 시스템
**담당 파일:**
- `src/briefing/generator.py`
- `src/briefing/sender.py`
- `scripts/run_briefing.py`

**generator.py:** 일간/주간/월간 기간 계산, 기간별 문서 + 키워드 검색, Gemini 요약, briefings 테이블 저장
**sender.py:** 브리핑 DB 저장 (push는 후순위)
**run_briefing.py:** CLI 인자로 briefing_type 받아 실행

---

## Step 4: 인프라 + 테스트 병렬 (2 에이전트)

### Agent G: 배포 설정
**담당 파일:**
- `.github/workflows/sync_data.yml` -- 30분 cron
- `.github/workflows/daily_briefing.yml` -- 일간/주간/월간 cron
- `Procfile`

### Agent H: 테스트
**담당 파일:**
- `tests/__init__.py`
- `tests/test_text_extractor.py`
- `tests/test_chunker.py`
- `tests/test_skill_handler.py` -- FastAPI TestClient + mock
- `tests/test_retriever.py`

---

## 검증 방법

1. `python -c "from src.config import settings"` -- config 로딩
2. 샘플 파일로 text_extractor + chunker 테스트
3. `python -c "from src.rag.embedder import Embedder; print(len(Embedder().embed('테스트')))"` → 768
4. `python scripts/init_db.py` → 테이블 확인
5. `python scripts/full_sync.py` → 동기화 실행
6. `uvicorn src.server.main:app` → /health, /skill/query 테스트
7. `python scripts/run_briefing.py daily` → 브리핑 생성
8. `pytest tests/` → 전체 테스트 통과
