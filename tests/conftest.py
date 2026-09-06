"""Isolate SYNCODE_HOME cho TOAN BO test suite — phai chay TRUOC khi import syncode
(vi config.py doc env var ngay luc import de tinh CONFIG_DIR)."""

import os
import tempfile

if "SYNCODE_HOME" not in os.environ:
    os.environ["SYNCODE_HOME"] = tempfile.mkdtemp(prefix="syncode-test-")
