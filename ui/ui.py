#!/usr/bin/env python3
# Browser UI for smoketesting a project and rendering ranges of its macroblocks.
# Usage: python3 ui/ui.py <ProjectName> [port]

import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

UI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(UI_DIR)
BUILD_DIR = os.path.join(ROOT, "build")
SAMPLERATE = 48000

if len(sys.argv) not in (2, 3):
    sys.exit("Usage: python3 ui/ui.py <ProjectName> [port]")
PROJECT = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) == 3 else 8765

matches = glob.glob(os.path.join(ROOT, "src/Projects/**", PROJECT + ".cpp"), recursive=True)
if not matches:
    sys.exit("ui.py: Project " + PROJECT + " does not exist.")
PROJECT_PATH = matches[0]
OUT_DIR = os.path.join(ROOT, "out", PROJECT)
MEDIA_DIR = os.path.join(ROOT, "media", PROJECT)
ACTIVE_PROJECT = os.path.join(ROOT, "src/Projects/.active_project.cpp")

state_lock = threading.Lock()
state = {"snapshot": None, "counts": [], "jobs": []}
jobs = queue.Queue()


def new_output_dir():
    name = time.strftime("%Y-%m-%d_%H.%M.%S")
    while os.path.exists(os.path.join(OUT_DIR, name)):
        time.sleep(0.1)
        name = time.strftime("%Y-%m-%d_%H.%M.%S")
    os.makedirs(os.path.join(OUT_DIR, name))
    return name


def run(cmd, cwd, log):
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    if subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
        raise RuntimeError("failed: " + " ".join(cmd))


def run_swaptube(binary, output_dir, args, log):
    # Swaptube resolves io_in/, io_out/, and MicroTeX relative to the build directory.
    for link, target in (("io_in", MEDIA_DIR), ("io_out", output_dir)):
        path = os.path.join(BUILD_DIR, link)
        if os.path.islink(path):
            os.unlink(path)
        os.symlink(target, path)
    try:
        run([binary] + args, BUILD_DIR, log)
    finally:
        os.unlink(os.path.join(BUILD_DIR, "io_in"))
        os.unlink(os.path.join(BUILD_DIR, "io_out"))


def smoke(job, out, log):
    os.makedirs(os.path.join(MEDIA_DIR, "latex"), exist_ok=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    shutil.copy(PROJECT_PATH, ACTIVE_PROJECT)
    try:
        run(["cmake", "-G", "Ninja", "..", "-DCOMPUTE_LANG="], BUILD_DIR, log)
        run(["ninja", "-j" + str(os.cpu_count())], BUILD_DIR, log)
    finally:
        shutil.move(ACTIVE_PROJECT, os.path.join(out, ".active_project.cpp"))
    # Renders use this copy, so they match the timeline this smoketest produced.
    binary = os.path.join(out, "swaptube")
    shutil.copy(os.path.join(BUILD_DIR, "swaptube"), binary)
    run_swaptube(binary, out, ["320", "180", str(job["fps"]), str(SAMPLERATE), "smoketest", "1", "0",
                               "io_out/microblock_counts.txt", "1", "all"], log)
    with open(os.path.join(out, "microblock_counts.txt")) as f:
        counts = [int(line) for line in f if line.strip()]
    with state_lock:
        state["snapshot"] = job["dir"]
        state["counts"] = counts


def render(job, out, log):
    snapshot = os.path.join(OUT_DIR, job["snapshot"])
    shutil.copy(os.path.join(snapshot, "microblock_counts.txt"), out)
    shutil.copy(os.path.join(snapshot, ".active_project.cpp"), out)
    run_swaptube(os.path.join(snapshot, "swaptube"), out,
                 [str(job["width"]), str(job["height"]), str(job["fps"]), str(SAMPLERATE), "render", "1", "0",
                  "io_out/microblock_counts.txt", "1", job["range"]], log)


def worker():
    while True:
        job = jobs.get()
        with state_lock:
            job["status"] = "running"
            job["dir"] = new_output_dir()
        out = os.path.join(OUT_DIR, job["dir"])
        try:
            with open(os.path.join(out, "log.txt"), "w") as log:
                (smoke if job["kind"] == "smoke" else render)(job, out, log)
            status = "done"
        except Exception as e:
            print("ui.py: job " + str(job["id"]) + " " + str(e))
            status = "failed"
        with state_lock:
            job["status"] = status


def add_job(**job):
    with state_lock:
        job.update(id=len(state["jobs"]), status="queued", dir=None)
        state["jobs"].append(job)
    jobs.put(job)


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.path = "/ui.html"
            self.directory = UI_DIR
            return super().do_GET()
        if self.path == "/api/state":
            with state_lock:
                return self.send_json(state)
        if self.path.startswith("/files/"):
            return self.send_file(self.path[len("/files/"):])
        self.send_error(404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            width, height, fps = int(req["width"]), int(req["height"]), int(req["fps"])
        except (ValueError, KeyError, TypeError):
            return self.send_error(400, "Expected width, height, and fps")
        if SAMPLERATE % fps != 0:
            return self.send_error(400, "Framerate must divide " + str(SAMPLERATE))
        if self.path == "/api/smoke":
            add_job(kind="smoke", fps=fps)
            return self.send_json({})
        if self.path == "/api/render":
            with state_lock:
                snapshot = state["snapshot"]
            if snapshot is None:
                return self.send_error(400, "Smoke first")
            add_job(kind="render", snapshot=snapshot, range="%d-%d" % (req["first"], req["end"]),
                    width=width, height=height, fps=fps)
            return self.send_json({})
        self.send_error(404)

    # Serves files under out/<Project>/, honoring Range requests so videos can seek.
    def send_file(self, rel):
        path = os.path.realpath(os.path.join(OUT_DIR, rel))
        if not path.startswith(os.path.realpath(OUT_DIR) + os.sep) or not os.path.isfile(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                start = size - int(match.group(2))
            end = min(end, size - 1)
            if start > end:
                return self.send_error(416)
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def log_message(self, *args):
        pass


threading.Thread(target=worker, daemon=True).start()
print("ui.py: Serving " + PROJECT + " at http://127.0.0.1:" + str(PORT))
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
