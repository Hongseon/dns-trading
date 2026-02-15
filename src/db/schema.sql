-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 통합 문서 테이블 (Dropbox + 메일 단일 인덱스)
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),

    -- 공통 메타데이터
    created_date TIMESTAMPTZ,
    updated_date TIMESTAMPTZ DEFAULT NOW(),

    -- Dropbox 메타데이터
    filename TEXT,
    folder_path TEXT,
    file_type TEXT,

    -- 이메일 메타데이터
    email_from TEXT,
    email_to TEXT,
    email_subject TEXT,
    email_date TIMESTAMPTZ,

    -- 청크 정보
    chunk_index INTEGER DEFAULT 0,

    UNIQUE(source_id, chunk_index)
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
    sync_type TEXT NOT NULL UNIQUE,
    last_cursor TEXT,
    last_sync_time TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 브리핑 히스토리
CREATE TABLE briefings (
    id BIGSERIAL PRIMARY KEY,
    briefing_type TEXT NOT NULL,
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
