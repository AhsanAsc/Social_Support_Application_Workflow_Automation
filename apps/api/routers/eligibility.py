from fastapi import APIRouter, Depends, HTTPException
from apps.api.deps import get_current_active_user, get_pg, get_mongo
from apps.api.schemas.decisions import EligibilityDecision
from domain.services.eligibility_engine import run_eligibility_engine

router = APIRouter(prefix="/eligibility", tags=["eligibility"])


@router.post("/evaluate/{application_id}", response_model=EligibilityDecision)
async def evaluate_application(
    application_id: int,
    user=Depends(get_current_active_user),
    pg=Depends(get_pg),
    mongo=Depends(get_mongo),
):
    """
    Run ML + rule-based eligibility engine.
    """
    decision = run_eligibility_engine(pg, mongo, application_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Application not found")

    return decision
