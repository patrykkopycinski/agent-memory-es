"""Thin HTTP client for the AMES REST API. The API key lives only in the
X-API-Key header of the underlying httpx client; it is never put in exception
text, repr(), or logs."""
from typing import Any, Optional

import httpx


class AmesError(Exception):
    """Error safe to show to the MCP client (never contains the API key)."""


class AmesClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0,
                 transport: Optional[httpx.BaseTransport] = None):
        if not api_key:
            raise AmesError("AMES_API_KEY is not set")
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  headers={"X-API-Key": api_key}, transport=transport)

    def __repr__(self) -> str:  # never leak the key via repr()
        return f"AmesClient(base_url={str(self._http.base_url)!r})"

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        try:
            r = self._http.request(method, path, json=body)
        except httpx.TimeoutException:
            raise AmesError("AMES backend timed out") from None
        except httpx.HTTPError as e:
            raise AmesError(f"AMES backend unreachable ({type(e).__name__})") from None
        if r.status_code == 401:
            raise AmesError("AMES rejected the API key (401); check AMES_API_KEY")
        if r.status_code >= 400:
            detail = ""
            try:
                detail = str(r.json().get("detail", ""))[:300]
            except Exception:
                pass
            raise AmesError(f"AMES backend error {r.status_code}: {detail}".rstrip(": "))
        return r.json()

    def recall(self, query: str, size: int, kinds: Optional[list] = None,
               as_of: Optional[str] = None) -> dict:
        body: dict = {"query": query, "size": size}
        if kinds:
            body["kinds"] = kinds
        if as_of:
            body["as_of"] = as_of
        return self._request("POST", "/memory/recall", body)

    def reflect(self, question: str) -> dict:
        return self._request("POST", "/memory/reflect", {"question": question})

    def stats(self) -> dict:
        return self._request("GET", "/memory/stats")

    def retain(self, kind: str, text: str, visibility: str,
               occurred_at: Optional[str] = None) -> dict:
        body: dict = {"kind": kind, "text": text, "visibility": visibility}
        if occurred_at:
            body["occurred_at"] = occurred_at
        return self._request("POST", "/memory/retain", body)

    def close(self) -> None:
        self._http.close()
