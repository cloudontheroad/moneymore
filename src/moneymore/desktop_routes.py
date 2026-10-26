from functools import lru_cache
from pathlib import Path
from typing import Literal

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


class ExportBody(BaseModel):
    filename: str = Field(min_length=1, max_length=180)
    kind: str = Field(pattern="^(positions|orders|trades)$")
    account_ref: str = Field(pattern="^[A-Za-z0-9_-]{1,64}$")
    trade_date: str = Field(pattern="^[0-9]{8}$")


class ReceiptLinkBody(BaseModel):
    intent_id: str = Field(pattern="^[a-f0-9]{32}$")
    snapshot_id: str = Field(pattern="^[a-f0-9]{64}$")
    broker_order_id: str = Field(pattern="^[A-Za-z0-9_-]{1,64}$")
    note: str = Field(min_length=1, max_length=200)
    acknowledge_observation_only: Literal[True]


class CaptureBody(BaseModel):
    allow_foreground_once: Literal[True]
    read_only_only: Literal[True]


@router.post("/capture")
def desktop_capture(body: CaptureBody, request: Request):
    require_local_action(request)
    try:
        return service().capture_service.start()
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.post("/capture/{job_id}/stop")
def desktop_capture_stop(job_id: str, request: Request):
    require_local_action(request)
    try:
        return service().capture_service.stop(job_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.post("/receipts/link")
def desktop_link_receipt(body: ReceiptLinkBody, request: Request):
    require_local_action(request)
    try:
        return service().bind_receipt(**body.model_dump(exclude={"acknowledge_observation_only"}))
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.post("/exports/import")
def desktop_import_export(body: ExportBody, request: Request):
    require_local_action(request)
    try:
        return service().import_exports(**body.model_dump())
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


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
