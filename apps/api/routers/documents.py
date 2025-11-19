from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from apps.api.deps import get_current_active_user, get_pg, get_mongo, get_qdrant
from apps.api.schemas.documents import DocumentRead
from services.ingestion.document_classifier import classify_document
from services.ingestion.pdf_parser import parse_pdf
from services.ingestion.image_ocr import parse_image
from services.ingestion.xlsx_parser import parse_xlsx
from services.persistence.postgres import DocumentRepository
from services.rag.vector_store import upsert_chunks
from services.rag.chunking import simple_text_chunk

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=DocumentRead)
async def upload_document(
    application_id: int = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_active_user),
    pg=Depends(get_pg),
    mongo=Depends(get_mongo),
    qdrant=Depends(get_qdrant),
):
    repo = DocumentRepository(pg)

    # 1. Save metadata
    doc_record = repo.create_document(application_id, file.filename)

    # 2. Classify (ID, bank statement, salary, resume, credit report, etc.)
    doc_type = classify_document(file.filename)
    repo.update_document_type(doc_record.id, doc_type)

    # 3. Parse content
    if file.filename.lower().endswith(".pdf"):
        text = parse_pdf(await file.read())
    elif file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        text = parse_image(await file.read())
    elif file.filename.lower().endswith(".xlsx"):
        text = parse_xlsx(await file.read())
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # 4. Save parsed content in MongoDB
    mongo["documents"].insert_one({
        "document_id": doc_record.id,
        "application_id": application_id,
        "doc_type": doc_type,
        "parsed_text": text,
    })

    # 5. Chunk + embed + index into Qdrant
    chunks = simple_text_chunk(text)
    upsert_chunks(qdrant, application_id, doc_record.id, chunks)

    return doc_record
