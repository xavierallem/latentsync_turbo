import json
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _NVML_AVAILABLE = False


class GPUMonitor:
    # Background sampler for GPU utilisation statistics via NVML.

    def __init__(self, device_index=None, interval=0.2):
        self.device_index = device_index
        self.interval = interval
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._handle = None

    def start(self):
        if not _NVML_AVAILABLE or not torch.cuda.is_available():
            return

        if self.device_index is None:
            self.device_index = torch.cuda.current_device()

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()

    def summary(self):
        if not self._samples:
            return None

        gpu_util = [sample[0] for sample in self._samples]
        mem_util = [sample[1] for sample in self._samples]
        mem_used = [sample[2] for sample in self._samples]

        return {
            "gpu_util_avg_percent": sum(gpu_util) / len(gpu_util),
            "gpu_util_max_percent": max(gpu_util),
            "mem_util_avg_percent": sum(mem_util) / len(mem_util),
            "mem_util_max_percent": max(mem_util),
            "mem_used_max_mib": max(mem_used),
            "samples_collected": len(self._samples),
        }

    def _run(self):
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append((float(util.gpu), float(util.memory), mem.used / (1024**2)))
            except Exception:
                break
            time.sleep(self.interval)


class TelemetrySession:
    # Collects Telemetry for Bennchmarking

    def __init__(self, device_index=None, interval=0.2):
        self._gpu_monitor = GPUMonitor(device_index=device_index, interval=interval)
        self._process = psutil.Process() if _PSUTIL_AVAILABLE else None
        self._rss_start = self._process.memory_info().rss / (1024**2) if self._process else None
        self._start_time = None
        self._end_time = None
        self._closed = False

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._gpu_monitor.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._closed:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._gpu_monitor.stop()
        self._end_time = time.perf_counter()
        self._closed = True

    def _ensure_closed(self):
        if not self._closed:
            self.__exit__(None, None, None)

    def metrics(self):
        if self._start_time is None:
            raise RuntimeError("TelemetrySession was not started")

        self._ensure_closed()
        elapsed = self._end_time - self._start_time if self._end_time is not None else 0.0

        metrics = {
            "elapsed_seconds": round(elapsed, 3),
        }

        if torch.cuda.is_available():
            peak_cuda = torch.cuda.max_memory_allocated() / (1024**2)
            metrics["peak_cuda_memory_mib"] = round(peak_cuda, 2)
        else:
            metrics["peak_cuda_memory_mib"] = None

        if self._process is not None:
            rss_end = self._process.memory_info().rss / (1024**2)
            metrics["process_ram_mib"] = round(rss_end, 2)
            if self._rss_start is not None:
                metrics["process_ram_delta_mib"] = round(rss_end - self._rss_start, 2)

        gpu_stats = self._gpu_monitor.summary()
        if gpu_stats is not None:
            for key, value in gpu_stats.items():
                metrics[key] = round(value, 2) if isinstance(value, float) else value
        else:
            metrics.update(
                {
                    "gpu_util_avg_percent": None,
                    "gpu_util_max_percent": None,
                    "mem_util_avg_percent": None,
                    "mem_util_max_percent": None,
                    "mem_used_max_mib": None,
                    "samples_collected": 0,
                }
            )

        return metrics

    def write_metrics(self, output_dir, base_name, extra=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = self.metrics()
        record = {
            "timestamp": timestamp,
            "metrics": metrics,
        }
        if extra:
            record.update(extra)
        metrics_path = output_dir / f"{base_name}_{timestamp}_metrics.json"
        metrics_path.write_text(json.dumps(record, indent=2))
        return metrics_path.as_posix(), metrics
