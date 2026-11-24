"""Launch separate Apollo trainers with a shared graceful stop request."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


def run_parallel(configs, common_args, train_script):
    def stream(band, process):
        with process.stdout:
            for line in process.stdout:
                if line.strip():
                    print(f"[{band}] {line.rstrip()}", flush=True)

    with tempfile.TemporaryDirectory(prefix="apollo-parallel-") as directory:
        inherited_stop = os.environ.get("APOLLO_STOP_FILE")
        stop_file = Path(inherited_stop or Path(directory) / "stop")
        env = {**os.environ, "APOLLO_STOP_FILE": str(stop_file), "PYTHONUNBUFFERED": "1"}
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        processes = {}
        readers = []
        reported = set()
        failed = False

        def stop(signum, frame):
            if stop_file.exists():
                for process in processes.values():
                    if process.poll() is None:
                        process.kill()
            else:
                stop_file.touch()
                print("\n[parallel] Stopping all specialists at their next safe point.", flush=True)

        previous = signal.signal(signal.SIGINT, stop)
        try:
            for band, config in configs.items():
                if stop_file.exists():
                    break
                command = [sys.executable, "-u", train_script, "--conf_dir", str(config), *common_args]
                process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors="replace", **group)
                processes[band] = process
                reader = threading.Thread(target=stream, args=(band, process), daemon=True)
                reader.start()
                readers.append(reader)

            while len(reported) < len(processes):
                for band, process in processes.items():
                    code = process.poll()
                    if code is None or band in reported:
                        continue
                    reported.add(band)
                    if code:
                        print(f"[parallel] {band} exited with status {code}.", flush=True)
                        failed = True
                        stop_file.touch()
                time.sleep(0.1)
        finally:
            signal.signal(signal.SIGINT, previous)
            for process in processes.values():
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdin.close()
            for reader in readers:
                reader.join()
    return int(failed)
