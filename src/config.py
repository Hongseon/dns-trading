from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Dropbox
    dropbox_access_token: str = ""
    dropbox_folder_path: str = "/업무"

    # Naver Mail
    naver_email: str = ""
    naver_password: str = ""
    naver_imap_server: str = "imap.naver.com"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_fallback_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # KakaoTalk
    kakao_admin_key: str = ""
    kakao_channel_id: str = ""

    # App constants
    embedding_dim: int = 768
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 10
    max_zip_depth: int = 2
    max_zip_extracted_size_mb: int = 50
    supported_extensions: list[str] = [
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".hwp", ".hwpx", ".cell",
        ".txt", ".csv", ".zip",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
