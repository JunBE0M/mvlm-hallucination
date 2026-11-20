"""재시작 가능한 결과 저장 유틸 (Vast interruptible 인스턴스 대비)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _encode(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    if v is None:
        return ""
    return v


class ResumableCsvWriter:
    """한 행씩 append, flush_every 행마다 flush+fsync.

    시작 시 기존 파일을 읽어 완전한 행만 남기고(중간에 끊겨 잘린 마지막 줄 제거) 다시 쓴 뒤 이어서 append.
    `done` 에 이미 처리된 key(qid) 집합이 들어 있다.
    """

    def __init__(self, path, columns: list[str], key: str = "qid", flush_every: int = 20):
        self.path = Path(path)
        self.columns = list(columns)
        self.key = key
        self.flush_every = max(1, flush_every)
        self.done: set[str] = set()
        self._pending = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover()
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.columns, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._sync()

    def _recover(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            raw = f.read()
        complete = raw if raw.endswith("\n") else raw[: raw.rfind("\n") + 1]
        rows = list(csv.reader(complete.splitlines()))
        if not rows:
            return
        header = rows[0]
        if header != self.columns:
            raise ValueError(f"{self.path}: existing columns differ from expected; use a new --out.\n"
                             f"existing={header}\nexpected={self.columns}")
        good = [r for r in rows[1:] if len(r) == len(header)]
        dropped = len(rows) - 1 - len(good)
        if dropped or complete != raw:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(good)
            os.replace(tmp, self.path)
            print(f"[resume] dropped {dropped} broken row(s) / partial line in {self.path}")
        k = header.index(self.key)
        self.done = {r[k] for r in good}

    def write(self, row: dict):
        self._w.writerow({c: _encode(row.get(c)) for c in self.columns})
        self.done.add(str(row[self.key]))
        self._pending += 1
        if self._pending >= self.flush_every:
            self._sync()

    def _sync(self):
        self._f.flush()
        os.fsync(self._f.fileno())
        self._pending = 0

    def close(self):
        if not self._f.closed:
            self._sync()
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def atomic_save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:     # 파일 핸들로 저장해야 numpy가 확장자를 붙이지 않음
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
