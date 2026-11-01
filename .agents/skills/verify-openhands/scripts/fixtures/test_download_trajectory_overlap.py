"""Safety contract for the real local trajectory receiver fixture."""

from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import unittest
import zipfile
import struct
import time
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading


spec = importlib.util.spec_from_file_location(
    "receiver", Path(__file__).with_name("download-trajectory-overlap.py")
)
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)


class ReceiverSafetyTest(unittest.TestCase):
    def args(self, origin="http://127.0.0.1:18810", extras=()):
        return ["--base-url", origin, "--conversation-id",
                "00000000-0000-0000-0000-000000000001", "--api-key-file",
                "/unused/private/session-key", "--output-dir", "/unused/private/output", *extras]

    def test_rejects_nonloopback_and_credential_bearing_origins_before_io(self):
        for origin in ("http://example.com:80", "http://localhost:80",
                       "http://127.0.0.1:80@evil.test:80", "https://127.0.0.1:80",
                       "http://127.0.0.1:80/?key=private", "http://127.0.0.1:80/#private"):
            with self.subTest(origin=origin), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    receiver.arguments(self.args(origin))
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(receiver.arguments(self.args()).clients, 3)
        self.assertEqual(receiver.arguments(self.args("http://[::1]:80")).base_url,
                         "http://[::1]:80")

    def test_rejects_unbounded_duration_concurrency_and_body_sizes(self):
        for extra in (("--seconds", "61"), ("--clients", "5"),
                      ("--rounds", "5001"), ("--slow-ms", "251"),
                      ("--max-bytes", str(17 * 1024 * 1024))):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    receiver.arguments(self.args(extras=extra))
                self.assertEqual(error.exception.code, 2)

    def test_a_broken_compressed_member_is_counted_without_aborting_receivers(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture.txt", "bounded fixture content" * 100)
        original = buf.getvalue()
        self.assertEqual(receiver.zip_outcome(original), "valid")
        damaged = bytearray(original)
        name_length, extra_length = struct.unpack_from("<HH", damaged, 26)
        start = 30 + name_length + extra_length
        damaged[start:start + 8] = b"\xff" * 8
        self.assertEqual(receiver.zip_outcome(damaged), "broken")

    def test_validation_budgets_and_deadlines_are_unverified(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture.txt", "x" * 100000)
            archive.writestr("second.txt", "small")
        body = buf.getvalue()
        self.assertEqual(receiver.zip_outcome(body, max_uncompressed_bytes=1024),
                         "unverified_uncompressed_limit")
        self.assertEqual(receiver.zip_outcome(body, max_members=1), "unverified_member_limit")
        self.assertEqual(receiver.zip_outcome(body, deadline=time.monotonic() - 1),
                         "unverified_deadline")

    def test_unsupported_compression_is_unverified(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("fixture.txt", "small")
        body = bytearray(buf.getvalue())
        struct.pack_into("<H", body, 8, 99)
        central = body.index(b"PK\x01\x02")
        struct.pack_into("<H", body, central + 10, 99)
        self.assertEqual(receiver.zip_outcome(body), "unverified_unsupported_archive")

    def test_intentional_network_cap_does_not_claim_transport_truncation(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("bounded.txt", "x" * 65536)
        body = archive.getvalue()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *_args):
                pass
        service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                key = Path(directory) / "key"
                key.write_text("dummy-test-only")
                args = receiver.arguments([
                    "--base-url", f"http://127.0.0.1:{service.server_port}",
                    "--conversation-id", "00000000-0000-0000-0000-000000000001",
                    "--api-key-file", str(key), "--output-dir", str(Path(directory) / "samples"),
                    "--seconds", "1", "--clients", "1", "--rounds", "1", "--max-bytes", "1024"])
                summary = asyncio.run(receiver.measure(args))
                self.assertEqual(summary["counts"]["unverified_network_byte_limit"], 1)
                self.assertNotIn("length_mismatch", summary["counts"])
                self.assertNotIn("valid_zip", summary["counts"])
                self.assertNotIn("broken_zip", summary["counts"])
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
