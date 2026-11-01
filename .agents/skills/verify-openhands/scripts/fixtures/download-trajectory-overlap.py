#!/usr/bin/env python3
"""Bounded real-loopback trajectory downloads; only sanitized counts are printed.

Run with the SDK environment's Python (which includes httpx). Raw ZIP samples
stay in a private output directory. No response is mocked or replayed.
"""

import argparse
import asyncio
from collections import Counter
import io
import ipaddress
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import UUID
import zipfile
import zlib


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--conversation-id", required=True)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2000, help="Maximum requests per client")
    p.add_argument("--slow-ms", type=float, default=0, help="Delay after each received chunk")
    p.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    p.add_argument("--max-uncompressed-bytes", type=int, default=16 * 1024 * 1024)
    p.add_argument("--max-members", type=int, default=512)
    a = p.parse_args(argv)
    u = urlsplit(a.base_url)
    try:
        local = ipaddress.ip_address(u.hostname or "").is_loopback
        port = u.port
    except ValueError:
        local, port = False, None
    if (u.scheme != "http" or not local or not port or u.username or u.password
            or u.query or u.fragment or u.path not in ("", "/")):
        p.error("base-url must be a literal loopback HTTP origin with a port")
    try:
        a.conversation_id = str(UUID(a.conversation_id))
    except ValueError:
        p.error("conversation-id must be a UUID")
    if not (1 <= a.seconds <= 60 and 1 <= a.clients <= 4
            and 1 <= a.rounds <= 5000 and 0 <= a.slow_ms <= 250
            and 1024 <= a.max_bytes <= 16 * 1024 * 1024
            and 1024 <= a.max_uncompressed_bytes <= 16 * 1024 * 1024
            and 1 <= a.max_members <= 512):
        p.error("bounds: seconds1..60, clients1..4, rounds1..5000, slow-ms0..250, max-bytes1KiB..16MiB")
    return a


def zip_outcome(body, *, max_uncompressed_bytes=16 * 1024 * 1024,
                max_members=512, deadline=float("inf")):
    try:
        if time.monotonic() >= deadline:
            return "unverified_deadline"
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) > max_members:
                return "unverified_member_limit"
            if sum(info.file_size for info in members) > max_uncompressed_bytes:
                return "unverified_uncompressed_limit"
            total = 0
            for info in members:
                if time.monotonic() >= deadline:
                    return "unverified_deadline"
                if info.is_dir():
                    continue
                with archive.open(info) as member:
                    while True:
                        if time.monotonic() >= deadline:
                            return "unverified_deadline"
                        chunk = member.read(min(32768, max_uncompressed_bytes - total + 1))
                        total += len(chunk)
                        if total > max_uncompressed_bytes:
                            return "unverified_uncompressed_limit"
                        if not chunk:
                            break
            return "valid"
    except (NotImplementedError, RuntimeError):
        return "unverified_unsupported_archive"
    except (zipfile.BadZipFile, OSError, EOFError, zlib.error):
        return "broken"


async def measure(a):
    import httpx

    key = a.api_key_file.read_text().strip()
    if not key:
        raise ValueError("The session-key file is empty")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(a.output_dir, 0o700)
    url = a.base_url.rstrip("/") + "/api/file/download-trajectory/" + a.conversation_id
    started = time.monotonic()
    deadline = started + a.seconds

    async def receiver(index):
        counts = Counter()
        samples = set()
        async with httpx.AsyncClient(
            headers={"X-Session-API-Key": key}, trust_env=False,
            follow_redirects=False, timeout=5,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            for _ in range(a.rounds):
                if time.monotonic() >= deadline:
                    break
                counts["attempts"] += 1
                body = bytearray()
                try:
                    async with client.stream("GET", url) as response:
                        counts["http_" + str(response.status_code)] += 1
                        abort_reason = None
                        async for chunk in response.aiter_bytes(chunk_size=32768):
                            if time.monotonic() >= deadline:
                                abort_reason = "unverified_deadline"
                                break
                            body.extend(chunk)
                            if len(body) > a.max_bytes:
                                abort_reason = "unverified_network_byte_limit"
                                break
                            if a.slow_ms:
                                await asyncio.sleep(a.slow_ms / 1000)
                        if response.status_code != 200:
                            continue
                        length = response.headers.get("content-length")
                        if abort_reason is None and length and int(length) != len(body):
                            counts["length_mismatch"] += 1
                        label = abort_reason or zip_outcome(
                            body, max_uncompressed_bytes=a.max_uncompressed_bytes,
                            max_members=a.max_members, deadline=deadline)
                        counts[label + "_zip" if label in ("valid", "broken") else label] += 1
                        counts["received_bytes"] += len(body)
                        if label not in samples:
                            raw = a.output_dir / f"client-{index}-{label}.zip"
                            raw.write_bytes(body)
                            os.chmod(raw, 0o600)
                            samples.add(label)
                except httpx.HTTPError as error:
                    counts["transport_" + type(error).__name__] += 1
        return counts

    combined = Counter()
    for counts in await asyncio.gather(*(receiver(i) for i in range(a.clients))):
        combined.update(counts)
    summary = {
        "origin": a.base_url.rstrip("/"), "clients": a.clients,
        "duration_limit_seconds": a.seconds, "round_limit_per_client": a.rounds,
        "slow_ms": a.slow_ms, "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_network_bytes": a.max_bytes, "max_uncompressed_bytes": a.max_uncompressed_bytes,
        "max_members": a.max_members,
        "counts": dict(combined), "raw_samples_private": True,
    }
    target = a.output_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    os.chmod(target, 0o600)
    return summary


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure(arguments())), indent=2))
