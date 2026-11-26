"""Process, phase and device measurements for an isolated V6 worker."""

import os
from pathlib import Path
import platform
import shutil
import subprocess
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np
import torch

from ..fitting.monitoring import peak_rss_bytes


RSS_INTERVAL_SECONDS = 0.02


def _proc_field(path, field):
    try:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                key, _, value = line.partition(":")
                if key.strip() == field:
                    return value.strip()
    except OSError:
        pass
    return None


def _proc_memory(key):
    value = _proc_field("/proc/meminfo", key)
    return int(value.split()[0]) * 1024 if value is not None else None


def _cgroup_memory(name):
    try:
        return int((Path("/sys/fs/cgroup") / name).read_text().strip())
    except (OSError, ValueError):
        return None


def _rss():
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _maximum(values):
    return max((value for value in values if value is not None), default=None)


def _driver_version():
    command = shutil.which("nvidia-smi")
    if command is None:
        return None
    try:
        result = subprocess.run(
            [command, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = result.stdout.splitlines()
    return versions[0].strip() if result.returncode == 0 and versions else None


def environment(device):
    gpu = None
    if device.type == "cuda":
        available, total = torch.cuda.mem_get_info(device)
        gpu = {
            "name": torch.cuda.get_device_name(device), "device": str(device),
            "available_memory_bytes": available, "total_memory_bytes": total,
            "compute_capability": torch.cuda.get_device_capability(device),
            "driver_version": _driver_version(),
        }
    return {
        "python": platform.python_version(), "numpy": np.__version__, "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "cpu_model": _proc_field("/proc/cpuinfo", "model name") or platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "system_ram_bytes": _proc_memory("MemTotal"), "available_system_ram_bytes": _proc_memory("MemAvailable"),
        "cgroup_ram_limit_bytes": _cgroup_memory("memory.max"),
        "cgroup_ram_usage_bytes": _cgroup_memory("memory.current"),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        )},
        "gpu": gpu, "pytorch_cuda_runtime": torch.version.cuda,
        "determinism": {
            "pytorch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "pytorch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


class CaseMeasurements:
    """Time exclusive phases and sample their resident memory independently."""

    def __init__(self):
        self.started = perf_counter()
        self.device = None
        self.baseline = _rss()
        self.phases = {}
        self.active = None
        self.cuda_measurement_error = None
        self._peak = None
        self._samples = 0
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._sample_phases, daemon=True)
        self._thread.start()

    def _sample(self):
        value = _rss()
        if value is not None:
            self._peak = _maximum((self._peak, value))
            self._samples += 1

    def _sample_phases(self):
        while not self._stop.wait(RSS_INTERVAL_SECONDS):
            with self._lock:
                if self.active is not None:
                    self._sample()

    @property
    def _cuda(self):
        return self.device is not None and self.device.type == "cuda" and self.cuda_measurement_error is None

    def start(self, name):
        self.end()
        if self._cuda:
            try:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError as error:
                self.cuda_measurement_error = str(error)
        with self._lock:
            self.active = (name, perf_counter())
            self._peak, self._samples = None, 0
            self._sample()

    def end(self):
        if self.active is None:
            return
        allocated = reserved = None
        if self._cuda:
            try:
                torch.cuda.synchronize(self.device)
                allocated = torch.cuda.max_memory_allocated(self.device)
                reserved = torch.cuda.max_memory_reserved(self.device)
            except RuntimeError as error:
                self.cuda_measurement_error = str(error)
        with self._lock:
            name, started = self.active
            seconds = perf_counter() - started
            self._sample()
            self.phases[name] = {
                "seconds": seconds, "sampled_peak_rss_bytes": self._peak, "rss_samples": self._samples,
                "cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved,
            }
            self.active = None

    def finish(self):
        try:
            self.end()
        finally:
            self._stop.set()
            self._thread.join()
        timing = {name: row["seconds"] for name, row in self.phases.items()}
        timing["total_case_seconds"] = perf_counter() - self.started
        peak = peak_rss_bytes()
        memory = {
            "baseline_rss_bytes": self.baseline, "peak_process_rss_bytes": peak,
            "peak_increase_above_baseline_bytes": max(0, peak - self.baseline)
            if peak is not None and self.baseline is not None else None,
            "cuda_peak_allocated_bytes": _maximum(row["cuda_peak_allocated_bytes"] for row in self.phases.values()),
            "cuda_peak_reserved_bytes": _maximum(row["cuda_peak_reserved_bytes"] for row in self.phases.values()),
            "cuda_measurement_error": self.cuda_measurement_error,
            "method": {
                "baseline_rss": "Linux /proc/self/statm before dataset and model loading",
                "peak_rss": "getrusage(RUSAGE_SELF) process-lifetime high-water mark, including imports",
                "phase_rss": "maximum sampled Linux /proc/self/statm RSS, including phase boundaries",
                "phase_rss_sample_interval_seconds": RSS_INTERVAL_SECONDS,
                "vram": "PyTorch allocator peaks reset at phase boundaries without clearing the allocator",
            },
            "phases": {name: {key: value for key, value in row.items() if key != "seconds"}
                       for name, row in self.phases.items()},
        }
        return timing, memory
