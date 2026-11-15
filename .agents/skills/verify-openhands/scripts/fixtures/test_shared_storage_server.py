"""Real child-group cleanup contract for the bounded SDK fixture."""

import importlib.util
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location(
    "shared_server", Path(__file__).with_name("shared-storage-server.py")
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@unittest.skipUnless(os.name == "posix", "The fixture requires POSIX process groups")
class GroupCleanupTest(unittest.TestCase):
    def check_orphan(self, ignore_term):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            worker = ("import signal,time,pathlib,sys; "
                      + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else "")
                      + "pathlib.Path(sys.argv[1]).touch(); time.sleep(20)")
            parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])"
            child = subprocess.Popen([sys.executable, "-c", parent, worker, str(ready)],
                                     start_new_session=True)
            try:
                child.wait(timeout=3)
                deadline = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                self.assertTrue(server.group_alive(child.pid))
                result = server.stop_owned_group(child, grace_seconds=.15, force_seconds=2)
                self.assertTrue(result["leader_stopped"])
                self.assertTrue(result["group_stopped"])
                self.assertEqual(result["forced"], ignore_term)
            finally:
                server.stop_owned_group(child, grace_seconds=.05, force_seconds=2)
                # Independent test cleanup also protects the deliberately failing
                # pre-fix regression run; assertions above test the helper itself.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_descendant_is_stopped_after_leader_already_exited(self):
        self.check_orphan(False)

    def test_ignoring_descendant_is_forced_after_bounded_grace(self):
        self.check_orphan(True)


if __name__ == "__main__":
    unittest.main()
