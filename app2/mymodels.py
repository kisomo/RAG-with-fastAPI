
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class FamilyName(str, Enum):
    Muthoka = "Muthoka"
    Kisomo = "Kisomo"

class QueryInput(BaseModel):
    children: str
    session_id: str = Field(default=None)
    model: FamilyName = Field(default=FamilyName.Muthoka)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: FamilyName
    max_size : int
    excess_no: int

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int


