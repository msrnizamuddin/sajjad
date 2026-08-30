"""Acquire official HIBA collection metadata without downloading images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


COLLECTION_ID = 251
EXPECTED_DOI = "10.34970/587329"
EXPECTED_TITLE = "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)"
EXPECTED_IMAGE_COUNT = 1616
DEFAULT_BASE_API_URL = "https://api.isic-archive.com/api/v2"
USER_AGENT = (
    "Skin-Cancer-Hierarchical-Classification/Phase10B "
    "(official HIBA metadata acquisition; research)"
)
DEFAULT_ROOT = Path("data/external/hiba")


class AcquisitionError(RuntimeError):
    """Raised when official metadata cannot be acquired safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(payload: Mapping[str, Any], paths: Sequence[str]) -> tuple[Any, str]:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return value, path
    return None, ""


def normalize_doi(value: Any) -> str:
    text = str(value).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.casefold().startswith(prefix):
            return text[len(prefix):].strip()
    return text


def verify_collection(payload: Mapping[str, Any]) -> dict[str, Any]:
    collection_id, id_path = _first(payload, ("id", "_id", "collectionId"))
    doi, doi_path = _first(payload, ("doi", "metadata.doi"))
    title, title_path = _first(payload, ("name", "title"))
    count, count_path = _first(
        payload, ("imageCount", "image_count", "images.count", "count")
    )
    try:
        parsed_id = int(collection_id)
    except (TypeError, ValueError) as exc:
        raise AcquisitionError("Collection response has no valid collection ID.") from exc
    # The ISIC Archive API's /collections/{id}/ endpoint has stopped returning any
    # image-count field (imageCount/image_count/images.count/count all absent as of
    # 2026); this collection-level count was only ever a pre-flight sanity check.
    # The authoritative check already happens later in collect_metadata(), which
    # compares the actual number of records retrieved from the paginated /images/
    # endpoint against EXPECTED_IMAGE_COUNT. So when the collection response omits
    # a count field entirely, skip this early check rather than fail acquisition on
    # a field the live API no longer sends; if the field IS present, still enforce it.
    parsed_count: int | None
    if count is None:
        parsed_count = None
    else:
        try:
            parsed_count = int(count)
        except (TypeError, ValueError) as exc:
            raise AcquisitionError("Collection response has an invalid image count.") from exc
        if parsed_count != EXPECTED_IMAGE_COUNT:
            raise AcquisitionError(
                "Expected image-count mismatch: "
                f"expected {EXPECTED_IMAGE_COUNT}, got {parsed_count}."
            )
    if parsed_id != COLLECTION_ID:
        raise AcquisitionError(
            f"Collection ID mismatch: expected {COLLECTION_ID}, got {parsed_id}."
        )
    if normalize_doi(doi) != EXPECTED_DOI:
        raise AcquisitionError(
            f"DOI mismatch: expected {EXPECTED_DOI}, got {doi!r}."
        )
    if str(title) != EXPECTED_TITLE:
        raise AcquisitionError(
            f"Title mismatch: expected {EXPECTED_TITLE!r}, got {title!r}."
        )
    return {
        "collection_id": parsed_id,
        "collection_id_source_path": id_path,
        "doi": str(doi),
        "doi_source_path": doi_path,
        "title": str(title),
        "title_source_path": title_path,
        "expected_image_count": parsed_count,
        "image_count_source_path": count_path,
    }


def ensure_under_root(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise AcquisitionError(f"Output path escapes HIBA root: {path}")
    return resolved


def official_urls(base_api_url: str) -> tuple[str, str]:
    base = base_api_url.rstrip("/")
    collection_url = f"{base}/collections/{COLLECTION_ID}/"
    query = urllib.parse.urlencode({
        "collections": COLLECTION_ID,
        "limit": 100,
        "offset": 0,
    })
    return collection_url, f"{base}/images/?{query}"


def fetch_json(
    url: str,
    *,
    timeout: float,
    max_attempts: int,
    backoff_seconds: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[Any, int]:
    if max_attempts < 1:
        raise AcquisitionError("max_attempts must be at least 1.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with opener(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status < 200 or status >= 300:
                    raise AcquisitionError(
                        f"Unexpected HTTP status {status} for {url}"
                    )
                return json.loads(response.read().decode("utf-8-sig")), status
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise AcquisitionError(
        f"Request failed after {max_attempts} attempts: {url}"
    ) from last_error


def _page_items(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(payload, Mapping):
        raise AcquisitionError("Image page must be a JSON object.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise AcquisitionError("Image page has no results list.")
    if any(not isinstance(item, dict) for item in results):
        raise AcquisitionError("Every image result must be a JSON object.")
    next_url = payload.get("next")
    if next_url is not None and not isinstance(next_url, str):
        raise AcquisitionError("Image page next value must be a URL or null.")
    return [dict(item) for item in results], next_url


def image_id(record: Mapping[str, Any]) -> str:
    value, _ = _first(record, ("isic_id", "isicId", "name", "_id", "id"))
    return "" if value is None else str(value).strip()


def _fixture_pages(directory: Path) -> list[tuple[str, Any, int]]:
    collection_path = directory / "collection.json"
    if not collection_path.is_file():
        raise AcquisitionError(f"Fixture collection is missing: {collection_path}")
    output = [
        (
            collection_path.as_uri(),
            json.loads(collection_path.read_text(encoding="utf-8-sig")),
            200,
        )
    ]
    page_number = 1
    while True:
        page_path = directory / f"page_{page_number}.json"
        if not page_path.exists():
            break
        output.append((
            page_path.as_uri(),
            json.loads(page_path.read_text(encoding="utf-8-sig")),
            200,
        ))
        page_number += 1
    if len(output) == 1:
        raise AcquisitionError("Fixture mode requires page_1.json.")
    return output


def collect_metadata(
    *,
    base_api_url: str = DEFAULT_BASE_API_URL,
    authorize_network: bool = False,
    fixture_directory: Path | None = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    fetcher: Callable[..., tuple[Any, int]] = fetch_json,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    request_log: list[dict[str, Any]] = []
    if fixture_directory is not None:
        responses = _fixture_pages(fixture_directory.resolve())
        collection_endpoint, collection_payload, collection_status = responses[0]
        page_responses = responses[1:]
    else:
        if not authorize_network:
            raise AcquisitionError(
                "Live network access refused. Pass "
                "--authorize-network-acquisition explicitly."
            )
        collection_endpoint, page_endpoint = official_urls(base_api_url)
        collection_payload, collection_status = fetcher(
            collection_endpoint,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        page_responses = []

    if not isinstance(collection_payload, Mapping):
        raise AcquisitionError("Collection response must be a JSON object.")
    verified = verify_collection(collection_payload)
    request_log.append({
        "endpoint_url": collection_endpoint,
        "requested_at_utc": utc_now(),
        "response_status": collection_status,
        "page_number": 0,
        "item_count": 1,
        "kind": "collection_metadata",
    })

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    page_number = 1
    while True:
        if fixture_directory is not None:
            if page_number > len(page_responses):
                raise AcquisitionError(
                    "Fixture pagination ended before a null next value."
                )
            endpoint, page_payload, status = page_responses[page_number - 1]
        else:
            print(f"Fetching image page {page_number} ({len(records)} images so far)...", flush=True)
            page_payload, status = fetcher(
                page_endpoint,
                timeout=timeout,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
            endpoint = page_endpoint
            # A brief pause between sequential page requests. The live API has been
            # observed to occasionally stall (accept the connection, never send a
            # response) after several rapid requests in a row; this is cheap
            # insurance against that, not a correctness requirement.
            time.sleep(0.25)
        items, next_url = _page_items(page_payload)
        request_log.append({
            "endpoint_url": endpoint,
            "requested_at_utc": utc_now(),
            "response_status": status,
            "page_number": page_number,
            "item_count": len(items),
            "kind": "collection_images",
        })
        for item in items:
            identifier = image_id(item)
            if not identifier:
                raise AcquisitionError(
                    f"Image record on page {page_number} has no image ID."
                )
            if identifier in seen:
                raise AcquisitionError(f"Duplicate ISIC image ID: {identifier}")
            seen.add(identifier)
            records.append(item)
        if next_url is None:
            break
        if fixture_directory is None:
            page_endpoint = urllib.parse.urljoin(endpoint, next_url)
        page_number += 1

    if len(records) != EXPECTED_IMAGE_COUNT:
        raise AcquisitionError(
            f"Retrieved image count mismatch: expected {EXPECTED_IMAGE_COUNT}, "
            f"got {len(records)}."
        )
    verified["retrieved_image_count"] = len(records)
    return dict(collection_payload), records, request_log


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def publish_acquisition(
    root: Path,
    collection: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    request_log: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    root = root.resolve()
    paths = {
        "collection": root / "source" / "collection_251.json",
        "attribution": root / "source" / "collection_251_attribution.json",
        "environment": root / "source" / "acquisition_environment.json",
        "request_log": root / "source" / "acquisition_request_log.json",
        "raw_jsonl": root / "metadata" / "collection_251_images.raw.jsonl",
    }
    for path in paths.values():
        ensure_under_root(path, root)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite finalized acquisition file(s): "
            + ", ".join(existing)
        )

    attribution, attribution_path = _first(
        collection, ("attribution", "metadata.attribution")
    )
    payloads: dict[str, Any] = {
        "collection": collection,
        "attribution": {
            "exact_value": attribution,
            "source_path": attribution_path,
            "missing": attribution is None,
        },
        "environment": {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "user_agent": USER_AGENT,
            "image_download_performed": False,
        },
        "request_log": list(request_log),
    }
    temporary: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for key, final_path in paths.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
            )
            os.close(descriptor)
            temporary[key] = Path(name)
        _write_json(temporary["collection"], payloads["collection"])
        _write_json(temporary["attribution"], payloads["attribution"])
        _write_json(temporary["environment"], payloads["environment"])
        _write_json(temporary["request_log"], payloads["request_log"])
        _write_jsonl(temporary["raw_jsonl"], records)
        raced = [str(path) for path in paths.values() if path.exists()]
        if raced:
            raise FileExistsError(
                "Acquisition output appeared during publication: " + ", ".join(raced)
            )
        for key, final_path in paths.items():
            os.rename(temporary[key], final_path)
            published.append(final_path)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-api-url", default=DEFAULT_BASE_API_URL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--authorize-network-acquisition", action="store_true")
    parser.add_argument("--fixture-directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixture_directory is not None and args.authorize_network_acquisition:
        raise AcquisitionError(
            "Fixture mode and network authorization are mutually exclusive."
        )
    if (
        args.fixture_directory is None
        and args.output_root.resolve() != DEFAULT_ROOT.resolve()
    ):
        raise AcquisitionError(
            "Live acquisition output root is fixed at data/external/hiba."
        )
    collection, records, request_log = collect_metadata(
        base_api_url=args.base_api_url,
        authorize_network=args.authorize_network_acquisition,
        fixture_directory=args.fixture_directory,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
    )
    verified = verify_collection(collection)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_validated_no_files_written",
            **verified,
            "retrieved_image_count": len(records),
        }, indent=2))
        return 0
    publish_acquisition(args.output_root, collection, records, request_log)
    print(f"Published metadata for {len(records)} images; no images downloaded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AcquisitionError, FileExistsError, OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
