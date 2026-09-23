"""Run a frozen training directory in a durable Linux process with an exclusive lock.

Usage: python run_guard.py DIRECTORY -- python launch.py
The launcher must run from a fresh version directory. Existing evidence is preserved.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.directory.resolve()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    lock = (root / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = root / 'frozen_run'
    snapshot.mkdir(exist_ok=False)
    files = list(root.glob('*.py')) + list(root.glob('protocol.json')) + list((root / 'prepared').glob('*.jsonl*'))
    manifest = {}
    for source in files:
        rel = source.relative_to(root)
        destination = snapshot / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        destination.write_bytes(content)
        destination.chmod(0o444)
        manifest[str(rel)] = hashlib.sha256(content).hexdigest()
    protocol_path=root/'protocol.json'
    external=json.loads(protocol_path.read_text()).get('external_inputs',[]) if protocol_path.exists() else []
    for index,name in enumerate(external):
        source=Path(name).resolve()
        content=source.read_bytes()
        destination=snapshot/'external'/f'{index:02d}_{source.name}'
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(content);destination.chmod(0o444)
        manifest[str(source)]=hashlib.sha256(content).hexdigest()
    write(snapshot / 'manifest.json', manifest)
    identity = {'pid': os.getpid(), 'command': command, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    stopped = []

    def stop(signum, frame):
        stopped.append(signum)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    termination_at = None
    changed = []
    while child.poll() is None:
        changed = [name for name, digest in manifest.items()
                   if not (root / name).exists() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
        if changed and not stopped:
            stop(signal.SIGTERM, None)
        if stopped:
            termination_at = termination_at or time.monotonic()
            if time.monotonic() - termination_at > 20:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        write(root / 'supervisor_status.json', {**identity, 'child_pid': child.pid, 'status': 'stopping' if stopped else 'running',
              'heartbeat_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'changed_inputs': changed})
        time.sleep(2)
    code = child.wait()
    result = {**identity, 'status': 'complete' if code == 0 and not stopped else 'interrupted' if stopped else 'failed',
              'exit_code': code, 'signals': stopped, 'changed_inputs': changed}
    write(root / 'supervisor_status.json', result)
    if result['status'] != 'complete':
        previous = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        write(root / 'status.json', {**previous, 'status': result['status'], 'supervisor': result})
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
