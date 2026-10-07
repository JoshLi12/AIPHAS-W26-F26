"""The live page reads flushed events, including a line that is still being written."""

from __future__ import annotations

import http.client
import socket
import tempfile
import unittest
from pathlib import Path

from standardized_pipeline.events import EventLog
from standardized_pipeline.live import read_new_events, start_live_view


class LiveViewTests(unittest.TestCase):
    def test_partial_line_waits_for_the_next_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_bytes(b'{"stage":"schema"}\n{"stage":"ing')
            lines, position = read_new_events(path, 0)
            self.assertEqual(lines, ['{"stage":"schema"}'])
            path.write_bytes(b'{"stage":"schema"}\n{"stage":"ingest"}\n')
            lines, position = read_new_events(path, position)
            self.assertEqual(lines, ['{"stage":"ingest"}'])
            self.assertGreater(position, 0)

    def test_replaced_log_is_read_from_the_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"stage":"schema","columns":["a","b","c"]}\n', encoding="utf-8")
            _, position = read_new_events(path, 0)
            path.write_text('{"stage":"ingest"}\n', encoding="utf-8")
            lines, _ = read_new_events(path, position)
            self.assertEqual(lines, ['{"stage":"ingest"}'])

    def test_page_receives_events_as_they_are_flushed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            view = start_live_view(output, 0)
            try:
                host, port = view.server.server_address[:2]
                page = http.client.HTTPConnection(host, port, timeout=2)
                page.request("GET", "/")
                body = page.getresponse().read().decode("utf-8")
                self.assertIn("Standardized pipeline", body)
                page.close()

                EventLog(output / "events.jsonl", reset=True).emit(
                    {"stage": "schema", "schema": "lab_observation", "threshold": 0.8, "llm": "offline"}
                )
                sock = socket.create_connection((host, port), timeout=2)
                sock.sendall(b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                received = b""
                while b"lab_observation" not in received:
                    chunk = sock.recv(256)
                    if not chunk:
                        break
                    received += chunk
                self.assertIn(b"data: ", received)
                self.assertIn(b"schema", received)
                sock.close()
            finally:
                view.stop()


if __name__ == "__main__":
    unittest.main()
