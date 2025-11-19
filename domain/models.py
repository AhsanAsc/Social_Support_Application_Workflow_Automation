from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class ApplicationStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    DECIDED = "decided"


class DocumentType(str, Enum):
    EMIRATES_ID = "emirates_id"
    BANK_STATEMENT = "bank_statement"
    SALARY_CERTIFICATE = "salary_certificate"
    UTILITY_BILL = "utility_bill"
    CREDIT_REPORT = "credit_report"
    ASSETS_LIABILITIES = "assets_liabilities"
    RESUME = "resume"
    OTHER = "other"


class Application(BaseModel):
    id: int | None = None
    applicant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ApplicationStatus = ApplicationStatus.DRAFT
    officer_id: Optional[str] = None


class Document(BaseModel):
    id: int | None = None
    application_id: int
    doc_type: DocumentType
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EligibilityLabel(str, Enum):
    ELIGIBLE = "eligible"
    BORDERLINE = "borderline"
    NOT_ELIGIBLE = "not_eligible"


class EligibilityDecision(BaseModel):
    application_id: int
    label: EligibilityLabel
    score: float
    reasons: List[str] = []


class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
