from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .desktop_execution import DesktopExecutionService

router = APIRouter(prefix="/api/desktop-execution")


@lru_cache(maxsize=1)
def service():
    return DesktopExecutionService(Path(__file__).resolve().parents[2])


def require_local_action(request: Request):
    allowed = {
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:8788",
        "http://localhost:8788",
    }
    if (
        request.headers.get("origin") not in allowed
        or request.headers.get("x-moneymore-desktop") != "prepare-v1"
    ):
        raise HTTPException(403, "仅接受MoneyMore本地网页发起的显式操作")


class PrepareBody(BaseModel):
    source_account: str = Field(min_length=1, max_length=160)


class ClientBody(BaseModel):
    executable: str = Field(min_length=1, max_length=500)


@router.post("/client")
def desktop_client(body: ClientBody, request: Request):
    require_local_action(request)
    try:
        return service().configure_client(body.executable)
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.get("")
def desktop_status():
    return service().snapshot()


@router.post("/probe")
def desktop_probe(request: Request):
    require_local_action(request)
    try:
        return service().probe()
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.post("/prepare")
def desktop_prepare(body: PrepareBody, request: Request):
    require_local_action(request)
    try:
        return service().prepare(body.source_account)
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
