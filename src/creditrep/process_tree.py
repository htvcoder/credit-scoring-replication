"""Bounded cleanup for process trees owned by multiprocessing supervisors."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import psutil


@dataclass(frozen=True)
class _Identity:
    process: psutil.Process
    pid: int
    create_time: float


class _CleanupFailure(RuntimeError):
    """Internal fail-closed cleanup error."""


def isolate_process_group() -> None:
    """Create a POSIX containment boundary before supervised workload code."""
    if os.name == "posix":
        os.setsid()


def _kill_owned_roots(processes: list[Any], timeout: float) -> None:
    """Best-effort fallback that still respects root reaping ownership."""
    seen: set[int] = set()
    for item in processes:
        pid = getattr(item, "pid", None)
        key = pid if isinstance(pid, int) else id(item)
        if key in seen:
            continue
        seen.add(key)
        try:
            if item.is_alive():
                item.kill()
            item.join(timeout=timeout)
        except (AssertionError, AttributeError, OSError, psutil.Error):
            continue


def _best_effort_failure_cleanup(
    processes: list[Any],
    descendants: dict[tuple[int, float], _Identity],
    timeout: float,
) -> None:
    """Remove every identity already proven in-scope before returning failure."""
    signalled: list[psutil.Process] = []
    for identity in descendants.values():
        try:
            if _same_process(identity):
                identity.process.kill()
                signalled.append(identity.process)
        except (psutil.NoSuchProcess, _CleanupFailure, OSError, psutil.Error):
            continue
    # Root-last remains mandatory even on the fail-closed path.
    _kill_owned_roots(processes, timeout)
    if signalled:
        try:
            psutil.wait_procs(signalled, timeout=timeout)
        except (AttributeError, OSError, psutil.Error):
            pass


def close_process_queue(queue: Any) -> bool:
    """Bounded parent-side queue teardown that cannot wait on a feeder thread."""
    try:
        cancel_join = getattr(queue, "cancel_join_thread", None)
        if cancel_join is not None:
            cancel_join()
        close = getattr(queue, "close", None)
        if close is not None:
            close()
        return True
    except (AttributeError, OSError, ValueError):
        return False


def _capture(process: psutil.Process) -> _Identity:
    return _Identity(process, process.pid, process.create_time())


def _same_process(identity: _Identity) -> bool:
    """Return false only when the original process identity is provably gone."""
    try:
        return identity.process.create_time() == identity.create_time
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError, psutil.Error) as exc:
        raise _CleanupFailure from exc


def _identity_is_gone(identity: _Identity) -> bool:
    """Use PID plus creation time, never bare PID existence, for verification."""
    try:
        current = psutil.Process(identity.pid)
        return current.create_time() != identity.create_time
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, OSError, psutil.Error) as exc:
        raise _CleanupFailure from exc


def terminate_and_reap(processes: list[Any], *, timeout: float = 2.0) -> bool:
    """Freeze and remove supervised trees without stealing root-child reaping.

    ``multiprocessing.Process`` owns each root and is therefore the only object
    used to reap it. Psutil is limited to identity-bound descendant discovery,
    suspension, signalling, waiting, and final verification. Once discovery
    starts, every live member is hard-killed: resuming a stopped root merely to
    deliver SIGTERM would open a window in which it could fork another child.
    """
    roots: dict[int, tuple[Any, _Identity]] = {}
    completed_roots: list[Any] = []
    descendants: dict[tuple[int, float], _Identity] = {}
    try:
        unverifiable_root = False
        for item in processes:
            pid = getattr(item, "pid", None)
            if not isinstance(pid, int) or pid <= 0:
                if getattr(item, "is_alive", lambda: False)():
                    try:
                        item.kill()
                        item.join(timeout=timeout)
                    except (AssertionError, AttributeError, OSError):
                        return False
                    unverifiable_root = True
                completed_roots.append(item)
                continue
            if pid in roots:
                continue
            # Synchronize a root that exited before cleanup before consulting
            # psutil. This preserves multiprocessing's waitpid ownership.
            item.join(timeout=0)
            if not item.is_alive():
                completed_roots.append(item)
                continue
            try:
                root = psutil.Process(pid)
                identity = _capture(root)
            except psutil.NoSuchProcess:
                completed_roots.append(item)
                continue
            except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                raise _CleanupFailure from exc
            root.suspend()
            if not _same_process(identity):
                completed_roots.append(item)
                continue
            roots[pid] = (item, identity)

        stable_scans = 0
        for _ in range(16):
            discovered = 0
            for _item, root_identity in roots.values():
                if not _same_process(root_identity):
                    continue
                try:
                    children = root_identity.process.children(recursive=True)
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                    raise _CleanupFailure from exc
                for child in children:
                    try:
                        identity = _capture(child)
                    except psutil.NoSuchProcess:
                        continue
                    except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                        raise _CleanupFailure from exc
                    key = (identity.pid, identity.create_time)
                    if identity.pid in roots or key in descendants:
                        continue
                    descendants[key] = identity
                    try:
                        child.suspend()
                    except psutil.NoSuchProcess:
                        continue
                    except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                        raise _CleanupFailure from exc
                    if _same_process(identity):
                        discovered += 1
            # POSIX supervised entries create a fresh session before workload
            # code. Group scans also catch descendants reparented during the
            # narrow enumerate/suspend interval; they retain the root's PGID.
            if os.name == "posix":
                contained_groups = {
                    identity.pid
                    for _item, identity in roots.values()
                    if _same_process(identity)
                    and os.getpgid(identity.pid) == identity.pid
                }
                if contained_groups:
                    for candidate in psutil.process_iter(attrs=["pid"]):
                        try:
                            if (
                                candidate.pid in roots
                                or os.getpgid(candidate.pid) not in contained_groups
                            ):
                                continue
                            identity = _capture(candidate)
                            key = (identity.pid, identity.create_time)
                            if key in descendants:
                                continue
                            descendants[key] = identity
                            candidate.suspend()
                            if _same_process(identity):
                                discovered += 1
                        except psutil.NoSuchProcess:
                            continue
                        except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                            raise _CleanupFailure from exc
            stable_scans = stable_scans + 1 if discovered == 0 else 0
            if stable_scans >= 2:
                break
        else:
            _best_effort_failure_cleanup(processes, descendants, timeout)
            return False

        # All descendants are signalled before any root. Hard kill is required
        # because every member is stopped and must never be resumed to run a
        # pending SIGTERM handler (which could create a descendant).
        waited_descendants: list[psutil.Process] = []
        for identity in descendants.values():
            if not _same_process(identity):
                continue
            try:
                identity.process.kill()
                waited_descendants.append(identity.process)
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                raise _CleanupFailure from exc

        # Root-last signalling and owner-handle-only reaping. In particular,
        # roots are never passed to psutil.wait()/wait_procs().
        for item, identity in roots.values():
            if _same_process(identity):
                try:
                    item.kill()
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass
                except (AttributeError, OSError, psutil.Error) as exc:
                    raise _CleanupFailure from exc
            item.join(timeout=timeout)
            if item.is_alive() or item.exitcode is None:
                _best_effort_failure_cleanup(processes, descendants, timeout)
                return False

        # Killing roots lets the OS adopt/reap any descendant zombies. Waiting
        # here is safe because these processes are never multiprocessing roots.
        if waited_descendants:
            _, alive = psutil.wait_procs(waited_descendants, timeout=timeout)
            for process in alive:
                identity = next(
                    value for value in descendants.values() if value.process is process
                )
                if _same_process(identity):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.AccessDenied, OSError, psutil.Error) as exc:
                        raise _CleanupFailure from exc
            if alive:
                _, alive = psutil.wait_procs(alive, timeout=timeout)
            if alive:
                _best_effort_failure_cleanup(processes, descendants, timeout)
                return False

        for item in completed_roots:
            try:
                item.join(timeout=timeout)
            except (AssertionError, AttributeError, OSError) as exc:
                raise _CleanupFailure from exc
            if item.is_alive() or (
                getattr(item, "pid", None) is not None and item.exitcode is None
            ):
                _best_effort_failure_cleanup(processes, descendants, timeout)
                return False
        success = (
            not unverifiable_root
            and all(_identity_is_gone(identity) for _item, identity in roots.values())
            and all(_identity_is_gone(identity) for identity in descendants.values())
        )
        if not success:
            _best_effort_failure_cleanup(processes, descendants, timeout)
        return success
    except (
        _CleanupFailure,
        AssertionError,
        AttributeError,
        OSError,
        ProcessLookupError,
        psutil.Error,
    ):
        _best_effort_failure_cleanup(processes, descendants, timeout)
        return False
