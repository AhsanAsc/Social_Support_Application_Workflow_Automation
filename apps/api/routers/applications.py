from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from apps.api.schemas.applications import ApplicationCreate, ApplicationRead
from apps.api.deps import get_current_active_user, get_pg
from services.persistence.postgres import ApplicationRepository

router = APIRouter(prefix="/applications", tags=["applications"])


@router.post("/", response_model=ApplicationRead, status_code=status.HTTP_201_CREATED)
async def create_application(
    payload: ApplicationCreate,
    user=Depends(get_current_active_user),
    pg=Depends(get_pg),
):
    repo = ApplicationRepository(pg)
    app_id = repo.create_application(payload, user)
    return repo.get_application_by_id(app_id)


@router.get("/", response_model=List[ApplicationRead])
async def list_applications(
    user=Depends(get_current_active_user),
    pg=Depends(get_pg),
):
    repo = ApplicationRepository(pg)
    return repo.list_applications(user)


@router.get("/{application_id}", response_model=ApplicationRead)
async def get_application(
    application_id: int,
    user=Depends(get_current_active_user),
    pg=Depends(get_pg),
):
    repo = ApplicationRepository(pg)
    result = repo.get_application_by_id(application_id)
    if not result:
        raise HTTPException(status_code=404, detail="Application not found")
    return result
