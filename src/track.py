"""실행 한 번의 걸린 시간과 VRAM(최대·평균)을 기록하는 래퍼.

    python -m src.track src.run_eval --model qwen2.5-vl-7b --questions ... --out ...
    -> src.run_eval 을 그대로 실행하고, 끝나면 COMPUTE_LOG(기본 results/coco2017_custom/compute/runs.csv)에 한 줄 추가

기록 열: start, end, elapsed_s, model, module, questions, out, gpu, vram_peak_gb, vram_mean_gb, exit_code, cmd
  - VRAM 은 이 파이썬 프로세스 안에서 PyTorch 가 잡은 GPU 메모리(torch.cuda.memory_reserved)다. 컨테이너(Vast)에서도
    프로세스별로 정확하고, GPU 한 장을 여러 프로세스가 같이 써도 각자 잡힌다. CUDA 기본 공간(약 0.3–0.5GB)은 빠지므로
    nvidia-smi 숫자보다 그만큼 작다.
  - 최대: torch.cuda.max_memory_reserved. 평균: 1초마다 잰 memory_reserved 의 평균 (0 보다 큰 값만, 모델을 올리기 전은 제외).
  - GPU 가 없으면 VRAM 열은 비운다. 실행이 예외로 끝나도 한 줄을 남기고(exit_code 1) 예외는 그대로 다시 던진다.
  - 여러 프로세스가 동시에 써도 줄이 섞이지 않게 파일 잠금(fcntl.flock)을 건다.
"""
from __future__ import annotations

import csv
import fcntl
import os
import runpy
import shlex
import sys
import threading
import time
from pathlib import Path

DEFAULT_LOG = "results/coco2017_custom/compute/runs.csv"
COLUMNS = ["start", "end", "elapsed_s", "model", "module", "questions", "out", "gpu",
           "vram_peak_gb", "vram_mean_gb", "exit_code", "cmd"]
INTERVAL_S = 1.0


def _arg(args: list[str], name: str) -> str:
    """args 에서 '--name 값' 의 값 (없으면 빈 문자열)."""
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return ""


def append_row(path: str | os.PathLike, row: dict) -> None:
    """파일 잠금을 건 상태로 한 줄 추가 (파일이 비어 있으면 머리줄부터)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if os.fstat(f.fileno()).st_size == 0:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class VramMeter:
    """1초마다 PyTorch 가 잡은 GPU 메모리를 재서 평균을 낸다 (값은 저장하지 않고 합과 개수만)."""

    def __init__(self, interval: float = INTERVAL_S):
        self.interval = interval
        self.total = 0.0
        self.count = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import torch
            self.torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self.torch = None

    def _reserved(self) -> int:
        t = self.torch
        return sum(t.cuda.memory_reserved(d) for d in range(t.cuda.device_count()))

    def _loop(self):
        while not self._stop.is_set():
            try:
                v = self._reserved()
            except Exception:
                v = 0
            if v > 0:
                self.total += v
                self.count += 1
            self._stop.wait(self.interval)

    def start(self):
        if self.torch is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.torch is None:
            return {"gpu": "", "vram_peak_gb": "", "vram_mean_gb": ""}
        t = self.torch
        peak = sum(t.cuda.max_memory_reserved(d) for d in range(t.cuda.device_count()))
        mean = self.total / self.count if self.count else 0.0
        return {"gpu": t.cuda.get_device_name(0), "vram_peak_gb": round(peak / 1024 ** 3, 3),
                "vram_mean_gb": round(mean / 1024 ** 3, 3)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0].startswith("-"):
        print("usage: python -m src.track <module> [args ...]", file=sys.stderr)
        return 2
    module, args = argv[0], argv[1:]
    log = os.environ.get("COMPUTE_LOG", DEFAULT_LOG)
    meter = VramMeter().start()
    t0 = time.time()
    code = 0
    saved_argv = sys.argv
    sys.argv = [module] + args
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        code = 1
        raise
    finally:
        t1 = time.time()
        sys.argv = saved_argv
        row = {"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
               "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
               "elapsed_s": round(t1 - t0, 1), "model": _arg(args, "--model"), "module": module,
               "questions": _arg(args, "--questions"), "out": _arg(args, "--out") or _arg(args, "--out_dir"),
               "exit_code": code, "cmd": shlex.join(["python", "-m", module] + args), **meter.stop()}
        append_row(log, row)
    return code


if __name__ == "__main__":
    sys.exit(main())
