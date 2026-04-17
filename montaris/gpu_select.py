"""Cross-platform GPU detection and preference setup.

This module MUST NOT import PySide6, vispy, or any GL library. It runs before
Qt is imported so its env-var mutations land before the GL driver is selected.

The goal is best-effort: we set the standard platform env vars that cause the
GL loader to prefer a discrete GPU. If the compositor/driver overrides us, the
runtime GL probe in view_3d.py will still report the truth in the UI.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional


class GpuInfo:
    __slots__ = ("index", "vendor", "name", "pci", "cuda_index")

    def __init__(self, index: int, vendor: str, name: str, pci: str = "",
                 cuda_index: Optional[int] = None):
        self.index = index
        self.vendor = vendor
        self.name = name
        self.pci = pci
        # For NVIDIA only: the 0-based ordinal from nvidia-smi -L, used by
        # CUDA_VISIBLE_DEVICES. Often different from the DRM card index.
        self.cuda_index = cuda_index

    def is_discrete(self) -> bool:
        if self.vendor in ("nvidia", "amd"):
            # Intel-made "Iris" is integrated even though it reports as Intel;
            # AMD integrated shows as "Radeon Graphics" usually. Name-based
            # hints: integrated APUs tend to contain "Graphics" at the end
            # rather than a model number. We keep it simple and trust the
            # vendor split — if the user has a real dGPU, it will be here.
            return True
        if self.vendor == "apple":
            return True  # M-series is unified but is the primary/only GPU
        return False

    def __repr__(self) -> str:
        return f"GpuInfo(index={self.index}, vendor={self.vendor!r}, name={self.name!r})"


# ---------- enumeration helpers ----------


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=timeout, text=True
        )
    except Exception:
        return ""


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""


def _vendor_from_pci_id(vendor_hex: str) -> str:
    v = (vendor_hex or "").lower().strip()
    return {
        "0x8086": "intel",
        "0x10de": "nvidia",
        "0x1002": "amd",
        "0x1022": "amd",
        "0x106b": "apple",
    }.get(v, "unknown")


def _lspci_detail(slot: str) -> str:
    if not shutil.which("lspci") or not slot:
        return ""
    out = _run(["lspci", "-s", slot])
    # "01:00.0 VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090] (rev a1)"
    m = re.search(r":\s*([^:]+)$", out.strip())
    if not m:
        return ""
    s = re.sub(r"\s*\(rev [^)]+\)\s*$", "", m.group(1)).strip()
    # Prefer the marketing name inside brackets
    m2 = re.search(r"\[([^\]]+)\]", s)
    if m2:
        return m2.group(1).strip()
    return s


def _list_nvidia_smi() -> List[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "-L"])
    gpus = []
    for line in out.splitlines():
        m = re.match(r"GPU (\d+):\s*(.+?)\s*\(UUID", line)
        if m:
            gpus.append(GpuInfo(index=int(m.group(1)), vendor="nvidia", name=m.group(2).strip()))
    return gpus


def _list_linux() -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    drm = "/sys/class/drm"
    if os.path.isdir(drm):
        for entry in sorted(os.listdir(drm)):
            if not re.fullmatch(r"card\d+", entry):
                continue
            idx = int(entry[4:])
            dev = os.path.join(drm, entry, "device")
            vendor_id = _read(os.path.join(dev, "vendor"))
            if not vendor_id:
                continue
            vendor = _vendor_from_pci_id(vendor_id)
            uevent = _read(os.path.join(dev, "uevent"))
            m = re.search(r"PCI_SLOT_NAME=(\S+)", uevent or "")
            slot = m.group(1) if m else ""
            name = _lspci_detail(slot) or f"{vendor.title()} GPU {idx}"
            gpus.append(GpuInfo(index=idx, vendor=vendor, name=name, pci=slot))
    # Refine NVIDIA entries with nvidia-smi names and record the smi ordinal
    # so CUDA_VISIBLE_DEVICES can pin the right GPU.
    smi = _list_nvidia_smi()
    if smi:
        smi_iter = iter(smi)
        for g in gpus:
            if g.vendor == "nvidia":
                try:
                    info = next(smi_iter)
                    g.name = info.name
                    g.cuda_index = info.index
                except StopIteration:
                    break
    return gpus


def _list_darwin() -> List[GpuInfo]:
    out = _run(["system_profiler", "SPDisplaysDataType"], timeout=5.0)
    gpus = []
    for i, name in enumerate(re.findall(r"Chipset Model:\s*(.+)", out)):
        nl = name.lower()
        vendor = (
            "apple" if "apple" in nl else
            "nvidia" if "nvidia" in nl else
            "amd" if ("amd" in nl or "radeon" in nl) else
            "intel" if "intel" in nl else
            "unknown"
        )
        gpus.append(GpuInfo(index=i, vendor=vendor, name=name.strip()))
    return gpus


def _list_windows() -> List[GpuInfo]:
    # Prefer PowerShell (wmic is deprecated on Win11).
    out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
    ], timeout=5.0)
    if not out.strip():
        out = _run(["wmic", "path", "win32_VideoController", "get", "name"], timeout=5.0)
    gpus = []
    for i, raw in enumerate(l for l in out.splitlines() if l.strip()):
        name = raw.strip()
        if name.lower() == "name":
            continue
        nl = name.lower()
        vendor = (
            "nvidia" if "nvidia" in nl else
            "amd" if ("amd" in nl or "radeon" in nl) else
            "intel" if "intel" in nl else
            "unknown"
        )
        gpus.append(GpuInfo(index=i, vendor=vendor, name=name))
    return gpus


def list_gpus() -> List[GpuInfo]:
    try:
        if sys.platform == "darwin":
            return _list_darwin()
        if sys.platform == "win32":
            return _list_windows()
        return _list_linux()
    except Exception:
        return []


def pick_discrete(gpus: List[GpuInfo]) -> Optional[GpuInfo]:
    for g in gpus:
        if g.vendor == "nvidia":
            return g
    for g in gpus:
        if g.vendor == "amd":
            return g
    for g in gpus:
        if g.vendor == "apple":
            return g
    return None


# ---------- env-var application ----------


_NVIDIA_EGL_VENDOR_PATHS = (
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "/etc/glvnd/egl_vendor.d/10_nvidia.json",
    "/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json",
)

STATUS_ENV_VAR = "MONTARIS_GPU_PREFERENCE"


def _set_env(key: str, value: str) -> None:
    # setdefault so users can still override manually from the shell.
    os.environ.setdefault(key, value)


def _apply_linux(chosen: GpuInfo) -> None:
    """Set Linux env vars that steer the GL loader toward ``chosen``.

    We intentionally do NOT set ``__EGL_VENDOR_LIBRARY_FILENAMES``. On Wayland,
    pinning the EGL ICD to nvidia-only makes GLVND refuse Mesa and EGL fails to
    initialize, which in turn tears down the Qt surface. Leaving the variable
    unset lets GLVND enumerate every vendor file in ``/usr/share/glvnd/`` and
    pick the one that can actually bind to the active display — which is what
    we want. ``__NV_PRIME_RENDER_OFFLOAD`` is the GLX lever; it's harmless on
    Wayland (ignored by the compositor) and effective on X11/XWayland.
    """
    if chosen.vendor == "nvidia":
        _set_env("__NV_PRIME_RENDER_OFFLOAD", "1")
        _set_env("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        cuda_idx = chosen.cuda_index if chosen.cuda_index is not None else 0
        _set_env("CUDA_VISIBLE_DEVICES", str(cuda_idx))
    elif chosen.vendor == "amd":
        _set_env("DRI_PRIME", "1")


def apply_selection(enable: bool, index: Optional[int]) -> dict:
    """Set env vars based on request. Returns a status dict.

    ``enable=False`` clears any offload hints so the compositor's default GPU
    (usually the integrated one) is used.

    Also writes a short human-readable summary to ``MONTARIS_GPU_PREFERENCE``
    so the UI can surface what was requested even from a different import site.
    """
    status = {
        "enabled": bool(enable),
        "index": index,
        "chosen": None,
        "reason": "",
        "platform": sys.platform,
        "gpus": list_gpus(),
    }

    if not enable:
        for k in (
            "__NV_PRIME_RENDER_OFFLOAD",
            "__GLX_VENDOR_LIBRARY_NAME",
            "CUDA_VISIBLE_DEVICES",
            "DRI_PRIME",
        ):
            os.environ.pop(k, None)
        status["reason"] = "GPU preference disabled (integrated)"
        os.environ[STATUS_ENV_VAR] = status["reason"]
        return status

    chosen: Optional[GpuInfo]
    if index is not None:
        gpus = status["gpus"]
        chosen = next((g for g in gpus if g.index == index), None)
        if chosen is None:
            status["reason"] = f"GPU index {index} not found; using default"
            os.environ[STATUS_ENV_VAR] = status["reason"]
            return status
    else:
        chosen = pick_discrete(status["gpus"])

    if chosen is None:
        status["reason"] = "No discrete GPU detected; using default"
        os.environ[STATUS_ENV_VAR] = status["reason"]
        return status

    status["chosen"] = chosen
    if sys.platform == "linux":
        _apply_linux(chosen)
    # macOS: Metal auto-selects; no reliable env-var lever.
    # Windows: driver app profiles rule — env vars have little effect.

    status["reason"] = f"Prefer {chosen.vendor.upper()}: {chosen.name}"
    os.environ[STATUS_ENV_VAR] = status["reason"]
    return status


def format_list(gpus: List[GpuInfo], chosen_index: Optional[int] = None) -> str:
    if not gpus:
        return "No GPUs detected."
    lines = ["Available GPUs:"]
    for g in gpus:
        tag = "discrete" if g.is_discrete() else "integrated" if g.vendor == "intel" else "other"
        marker = "  ← selected" if chosen_index is not None and g.index == chosen_index else ""
        lines.append(f"  [{g.index}] {g.name}  ({g.vendor}, {tag}){marker}")
    return "\n".join(lines)
