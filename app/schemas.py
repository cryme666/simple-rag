from pydantic import BaseModel, Field, HttpUrl


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[ChatMessage] = []


class SourceInfo(BaseModel):
    source: str
    source_type: str


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceInfo] = Field(default_factory=list)


class IngestFileResponse(BaseModel):
    message: str
    chunks_stored: int
    source: str


class ScrapeRequest(BaseModel):
    url: HttpUrl
    overwrite: bool = False


class ScrapeResponse(BaseModel):
    message: str
    chunks_stored: int
    source: str
    title: str
