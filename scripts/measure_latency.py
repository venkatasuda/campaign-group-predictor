"""Measure end-to-end prediction latency against a running service.

    python scripts/measure_latency.py --url http://localhost:8000
    python scripts/measure_latency.py --url https://campaign-api-....run.app --requests 200

Why this exists
---------------
`docs/architecture.md` states an expected p95 under 100 ms. Until this is run that is an
assertion, not a measurement, and an unverified performance number in an architecture
document is exactly the kind of claim a reviewer checks. This script replaces it with an
observation, or forces the claim to be corrected.

What it measures, and what it does not
--------------------------------------
It records **client-observed wall-clock latency**: serialisation, network, queueing,
inference and deserialisation together. That is the number a caller experiences, which is
the one worth quoting.

Against Cloud Run it is *not* a clean measure of service latency. It includes internet
round-trip from wherever this runs, and a scale-to-zero service pays a cold start on the
first request - which is why the first result is reported separately rather than folded
into the percentiles. Run it locally to isolate the application, and against Cloud Run to
see what a caller actually gets.

Percentiles, not the mean. A mean hides the tail, and the tail is what times out.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import (  # noqa: E402
    COMPARISON_FEATURES,
    GROUP_1_FEATURES,
    GROUP_2_FEATURES,
)


def build_payload() -> dict[str, dict[str, float]]:
    """A schema-valid request built from the column contract, not a hardcoded blob.

    Derived from `src.constants` so the payload cannot drift out of sync with the schema:
    if a feature is added, this fails loudly at the API's validation layer rather than
    silently measuring a rejected request.
    """
    return {
        "group_1": dict.fromkeys(GROUP_1_FEATURES, 0.5),
        "group_2": dict.fromkeys(GROUP_2_FEATURES, 0.4),
        "comparison": dict.fromkeys(COMPARISON_FEATURES, 0.1),
    }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Explicit because numpy's interpolation differs subtly."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(round(fraction * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Service base URL.")
    parser.add_argument("--requests", type=int, default=100, help="Requests to send.")
    parser.add_argument("--warmup", type=int, default=5, help="Requests excluded from statistics.")
    parser.add_argument("--endpoint", default="/predict", choices=["/predict", "/health"])
    parser.add_argument("--out", default="reports/latency.json")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    payload = build_payload()
    session = requests.Session()  # reuse the TCP/TLS connection; otherwise the handshake dominates

    print(f"Target:   {base}{args.endpoint}")
    print(f"Requests: {args.requests} (plus {args.warmup} warm-up)\n")

    durations: list[float] = []
    cold_start_ms: float | None = None
    failures = 0

    for index in range(args.requests + args.warmup):
        started = time.perf_counter()
        try:
            if args.endpoint == "/health":
                response = session.get(f"{base}/health", timeout=30)
            else:
                response = session.post(f"{base}/predict", json=payload, timeout=30)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if response.status_code != 200:
                failures += 1
                if failures <= 3:
                    print(f"  HTTP {response.status_code}: {response.text[:200]}")
                continue
        except requests.RequestException as error:
            failures += 1
            if failures <= 3:
                print(f"  request failed: {error}")
            continue

        # The first request against a scale-to-zero service pays container start plus
        # model deserialisation. Averaging it in would misrepresent steady-state latency;
        # hiding it entirely would misrepresent what the first caller of the day sees.
        if index == 0:
            cold_start_ms = elapsed_ms
        if index >= args.warmup:
            durations.append(elapsed_ms)

    if not durations:
        print("\nNo successful requests - is the service running?", file=sys.stderr)
        return 1

    results: dict[str, Any] = {
        "url": base,
        "endpoint": args.endpoint,
        "n_requests": len(durations),
        "n_failures": failures,
        "first_request_ms": round(cold_start_ms, 1) if cold_start_ms else None,
        "min_ms": round(min(durations), 1),
        "median_ms": round(statistics.median(durations), 1),
        "mean_ms": round(statistics.fmean(durations), 1),
        "p95_ms": round(percentile(durations, 0.95), 1),
        "p99_ms": round(percentile(durations, 0.99), 1),
        "max_ms": round(max(durations), 1),
    }

    width = max(len(key) for key in results)
    for key, value in results.items():
        print(f"  {key:<{width}}  {value}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")

    if failures:
        print(f"\n{failures} request(s) failed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
