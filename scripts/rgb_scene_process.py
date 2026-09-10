"""Linux owned-process supervision with finite cleanup, including detached descendants.

The supervisor becomes a subreaper. Only descendants of this invocation are
signaled/reaped, verified by /proc start time; unrelated user jobs are untouched.
"""
import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

OUTER_TIMEOUT_S = 1100
RECONSTRUCTION_BUDGET_S = 1000
CLEANUP_TIMEOUT_S = 5


class OwnedProcessFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"{record['phase']}: {record.get('error', 'nonzero exit')}; log={record['log']}")


class _Cancelled(Exception):
    pass


def _processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def _descendants(pid, table):
    owned = {pid}
    for _ in range(len(table)):
        added = {p for p, (parent, _, _) in table.items() if parent in owned}
        if added <= owned:
            break
        owned |= added
    return {p: table[p][1] for p in owned if p in table}


def _cleanup(proc, owned, timeout):
    table = _processes()
    owned.update(_descendants(proc.pid, table))
    # Also catch reparented, session-detached descendants: with a subreaper they
    # belong to this supervisor, but exclude all pre-existing children at launch.
    deadline = time.monotonic() + timeout
    statuses = {}
    # The session/process group was created by this invocation, never inherited
    # from the user's shell. Kill it as a group; pidfds below also cover owned
    # descendants which created a separate session.
    if proc.pid in table and owned.get(proc.pid) == table[proc.pid][1]:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    with selectors.DefaultSelector() as selector:
        for pid, birth in owned.items():
            if pid not in table or table[pid][1] != birth:
                statuses[str(pid)] = 'exited'
                continue
            try:
                fd = os.pidfd_open(pid)
                current = _processes().get(pid)
                if current is None or current[1] != birth:
                    os.close(fd)
                    statuses[str(pid)] = 'exited'
                    continue
                selector.register(fd, selectors.EVENT_READ, pid)
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                statuses[str(pid)] = 'exited'
        # One bounded readiness wait per owned process; no polling/idle loops.
        for _ in range(len(selector.get_map())):
            ready = selector.select(max(0, deadline - time.monotonic()))
            if not ready:
                break
            for key, _ in ready:
                pid = key.data
                try:
                    if pid == proc.pid:
                        proc.wait(timeout=0)
                        waited = pid
                    else:
                        waited, _ = os.waitpid(pid, os.WNOHANG)
                    statuses[str(pid)] = 'reaped' if waited else 'exited_not_reaped'
                except ChildProcessError:
                    statuses[str(pid)] = 'exited_reaped_by_owner'
                selector.unregister(key.fd)
                os.close(key.fd)
        for key in list(selector.get_map().values()):
            statuses[str(key.data)] = 'unresolved_after_SIGKILL'
            os.close(key.fd)
    proc.poll()
    return {'status': 'unresolved' if any('unresolved' in s or s == 'exited_not_reaped' for s in statuses.values()) else 'stopped',
            'processes': statuses, 'automatic_retry_allowed': False}


def run_owned(phase, argv, *, cwd, env, log, timeout, cleanup_timeout=CLEANUP_TIMEOUT_S):
    """Return exact execution record or raise with that record after cleanup."""
    started = time.monotonic()
    record = {'phase': phase, 'argv': list(map(str, argv)), 'timeout_s': timeout,
              'log': str(log), 'returncode': None, 'cleanup': {'status': 'not_spawned'}}
    proc = None
    previous = signal.getsignal(signal.SIGTERM)
    def cancel(signum, frame):
        raise _Cancelled('outer supervision cancelled nested command')
    # Linux runtime contract; failure here is explicit, not unsafe best effort.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('cannot enable owned child subreaper')
    baseline = _descendants(os.getpid(), _processes())
    owned = {}
    signal.signal(signal.SIGTERM, cancel)
    try:
        if timeout <= 0:
            raise subprocess.TimeoutExpired(record['argv'], timeout)
        with Path(log).open('x') as handle:
            proc = subprocess.Popen(record['argv'], cwd=cwd, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record['returncode'] = proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, _Cancelled) as exc:
                record['error'] = 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'outer_cancelled'
                owned = _descendants(proc.pid, _processes())
                # An outer timer gives its nested wrapper a cleanup margin.
                # Cancellation inside that wrapper instead proceeds directly to
                # owned SIGKILL/reaping so it can persist before the margin ends.
                if isinstance(exc, subprocess.TimeoutExpired):
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=cleanup_timeout + 1)
                    except subprocess.TimeoutExpired:
                        pass
            table = _processes()
            protected = set(baseline)
            for pid in baseline:
                if pid != os.getpid():
                    protected.update(_descendants(pid,table))
            owned.update({p: birth for p, birth in _descendants(os.getpid(), table).items()
                          if p not in protected})
            record['cleanup'] = _cleanup(proc, owned, cleanup_timeout)
            record['returncode'] = proc.returncode
            if record['cleanup']['status'] != 'stopped':
                record.setdefault('error', 'unresolved owned descendants')
            if proc.returncode:
                record.setdefault('error', 'nonzero exit')
    except subprocess.TimeoutExpired:
        record['error'] = 'deadline_exhausted_before_spawn'
    except OSError as exc:
        record['error'] = f'spawn/IO error: {exc}'
        if proc is not None:
            record['cleanup'] = _cleanup(proc, owned, cleanup_timeout)
    finally:
        signal.signal(signal.SIGTERM, previous)
        record['elapsed_s'] = time.monotonic() - started
    if 'error' in record:
        raise OwnedProcessFailure(record)
    return record
