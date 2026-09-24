"""Run a command with a duty-cycled CPU/GPU load and GPU telemetry (Windows).

The child process is suspended and resumed periodically with NtSuspendProcess /
NtResumeProcess, so average power draw drops while every computation is left
exactly as it was: results are identical, only wall time grows. Queued GPU work
drains during a pause; nothing is killed or restarted.

Telemetry (power, temperature, utilisation, SM clock) is appended to a CSV every
few seconds and fsynced, so the last readings survive a hard power loss.

Written because this laptop loses power under sustained GPU load
(2026-09-17: three Kernel-Power 41 events with BugcheckCode 0, on AC, 100% battery).

Usage:
  python experiments/run_throttled.py --on 1.0 --off 1.0 --log FILE -- <command ...>
"""
import argparse, ctypes, os, subprocess, sys, threading, time
from ctypes import wintypes

PROCESS_SUSPEND_RESUME = 0x0800
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
kernel32.OpenProcess.restype = wintypes.HANDLE


def _handle(pid):
    h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    return h


def telemetry(path, stop, every):
    q = "timestamp,power.draw,temperature.gpu,utilization.gpu,clocks.sm,memory.used"
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("wall," + q + "\n")
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception as e:  # telemetry must never stop the run
                out = f"telemetry-error {e!r}"
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{out}\n")
            f.flush()
            os.fsync(f.fileno())
            stop.wait(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=float, default=1.0, help="seconds running per cycle")
    ap.add_argument("--off", type=float, default=1.0, help="seconds suspended per cycle")
    ap.add_argument("--log", required=True, help="telemetry CSV")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    stop = threading.Event()
    t = threading.Thread(target=telemetry, args=(a.log, stop, a.every), daemon=True)
    t.start()
    child = subprocess.Popen(cmd)
    h = _handle(child.pid)
    suspended = False
    try:
        while child.poll() is None:
            time.sleep(a.on)
            if a.off > 0 and child.poll() is None:
                ntdll.NtSuspendProcess(h); suspended = True
                time.sleep(a.off)
                ntdll.NtResumeProcess(h); suspended = False
    finally:
        if suspended:
            ntdll.NtResumeProcess(h)
        kernel32.CloseHandle(h)
        stop.set()
        t.join(timeout=15)
    sys.exit(child.wait())


if __name__ == "__main__":
    main()
