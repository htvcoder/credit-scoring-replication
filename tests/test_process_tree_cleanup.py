from __future__ import annotations

from multiprocessing import get_all_start_methods, get_context
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest

from creditrep import process_tree
from creditrep.experiments import p7c4b2b_preflight as b2b
from creditrep.experiments import p7c4b2c_preflight as b2c


def _deep_sleeper(queue, depth: int) -> None:
    queue.put(os.getpid())
    if depth:
        context = get_context("spawn")
        child = context.Process(target=_deep_sleeper, args=(queue, depth - 1))
        child.start()
    time.sleep(60)


def _ignore_term() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


def _root_with_reparented_group_member(marker: str) -> None:
    process_tree.isolate_process_group()
    code = (
        "from pathlib import Path; import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "Path(sys.argv[1]).write_text(str(p.pid),encoding='utf-8')"
    )
    subprocess.Popen(
        [sys.executable, "-c", code, marker],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(60)


def _identity_gone(pid: int, create_time: float) -> bool:
    try:
        return psutil.Process(pid).create_time() != create_time
    except psutil.NoSuchProcess:
        return True


def test_shared_cleanup_contract_is_used_by_b2b_and_b2c():
    assert b2b.terminate_and_reap is process_tree.terminate_and_reap
    assert b2c.terminate_and_reap is process_tree.terminate_and_reap


@pytest.mark.parametrize("start_method", get_all_start_methods())
def test_available_start_methods_owner_reap_root(start_method):
    process = get_context(start_method).Process(target=time.sleep, args=(60,))
    process.start()
    assert process_tree.terminate_and_reap([process])
    assert not process.is_alive()
    assert process.exitcode is not None


def test_root_is_owner_reaped_and_never_passed_to_psutil_wait(monkeypatch):
    context = get_context("spawn")
    process = context.Process(target=time.sleep, args=(60,))
    process.start()
    pid = process.pid
    created = psutil.Process(pid).create_time()
    seen: list[set[int]] = []
    original = process_tree.psutil.wait_procs

    def recording_wait(items, timeout):
        seen.append({item.pid for item in items})
        return original(items, timeout=timeout)

    monkeypatch.setattr(process_tree.psutil, "wait_procs", recording_wait)
    assert process_tree.terminate_and_reap([process])
    assert not process.is_alive()
    assert process.exitcode is not None
    assert _identity_gone(pid, created)
    assert all(pid not in waited for waited in seen)


def test_pre_suspended_root_and_three_level_tree_are_removed():
    context = get_context("spawn")
    queue = context.Queue()
    root = context.Process(target=_deep_sleeper, args=(queue, 2))
    root.start()
    pids = {queue.get(timeout=10) for _ in range(3)}
    identities = {pid: psutil.Process(pid).create_time() for pid in pids}
    psutil.Process(root.pid).suspend()
    assert process_tree.terminate_and_reap([root])
    assert not root.is_alive()
    assert root.exitcode is not None
    assert all(_identity_gone(pid, created) for pid, created in identities.items())
    queue.close()
    queue.join_thread()


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM behavior")
def test_sigterm_ignoring_root_is_bounded_hard_killed():
    process = get_context("spawn").Process(target=_ignore_term)
    process.start()
    started = time.monotonic()
    assert process_tree.terminate_and_reap([process], timeout=1.0)
    assert time.monotonic() - started < 5.0
    assert process.exitcode is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group containment")
def test_reparented_descendant_cannot_escape_isolated_group(tmp_path):
    marker = tmp_path / "orphan-pid.txt"
    root = get_context("spawn").Process(
        target=_root_with_reparented_group_member, args=(str(marker),)
    )
    root.start()
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    leaf_pid = int(Path(marker).read_text(encoding="utf-8"))
    leaf_created = psutil.Process(leaf_pid).create_time()
    time.sleep(0.2)
    assert process_tree.terminate_and_reap([root])
    assert _identity_gone(leaf_pid, leaf_created)


def test_root_that_exits_before_cleanup_is_joined():
    process = get_context("spawn").Process(target=time.sleep, args=(0.01,))
    process.start()
    process.join(timeout=5)
    assert process_tree.terminate_and_reap([process])
    assert not process.is_alive()
    assert process.exitcode == 0


class _FakePsProcess:
    def __init__(
        self,
        pid=41,
        created=1.0,
        *,
        children=None,
        suspend_error=None,
        children_error=None,
        kill_error=None,
    ):
        self.pid = pid
        self.created = created
        self.alive = True
        self._children = children or []
        self.suspend_error = suspend_error
        self.children_error = children_error
        self.kill_error = kill_error
        self.kill_count = 0

    def create_time(self):
        if not self.alive:
            raise psutil.NoSuchProcess(self.pid)
        return self.created

    def suspend(self):
        if self.suspend_error is not None:
            raise self.suspend_error

    def children(self, recursive):
        assert recursive
        if self.children_error is not None:
            raise self.children_error
        return list(self._children)

    def kill(self):
        self.kill_count += 1
        if self.kill_error is not None:
            raise self.kill_error
        self.alive = False


class _FakeOwner:
    def __init__(self, process: _FakePsProcess, *, kill_works=True):
        self.pid = process.pid
        self.exitcode = None
        self.process = process
        self.kill_works = kill_works
        self.kill_count = 0

    def is_alive(self):
        return self.exitcode is None

    def kill(self):
        self.kill_count += 1
        if self.kill_works:
            self.process.alive = False
            self.exitcode = -9

    def join(self, timeout=None):
        del timeout


def test_pid_reuse_is_classified_gone_without_signalling_replacement(monkeypatch):
    original = _FakePsProcess()
    replacement = _FakePsProcess(created=2.0)
    owner = _FakeOwner(original)

    def process_for_pid(_pid):
        return original if original.alive else replacement

    monkeypatch.setattr(process_tree.psutil, "Process", process_for_pid)
    assert process_tree.terminate_and_reap([owner])
    assert owner.kill_count == 1
    assert replacement.kill_count == 0


def test_duplicate_root_is_not_double_signalled(monkeypatch):
    root = _FakePsProcess()
    replacement = _FakePsProcess(created=2.0)
    owner = _FakeOwner(root)
    monkeypatch.setattr(
        process_tree.psutil,
        "Process",
        lambda _pid: root if root.alive else replacement,
    )
    assert process_tree.terminate_and_reap([owner, owner])
    assert owner.kill_count == 1


@pytest.mark.parametrize(
    "error",
    [psutil.AccessDenied(41), OSError("injected cleanup failure")],
)
def test_root_suspend_errors_fail_closed(monkeypatch, error):
    root = _FakePsProcess(suspend_error=error)
    owner = _FakeOwner(root)
    monkeypatch.setattr(process_tree.psutil, "Process", lambda _pid: root)
    assert not process_tree.terminate_and_reap([owner])
    assert owner.kill_count == 1
    assert not owner.is_alive()


def test_root_join_timeout_fails_closed(monkeypatch):
    root = _FakePsProcess()
    owner = _FakeOwner(root, kill_works=False)
    monkeypatch.setattr(process_tree.psutil, "Process", lambda _pid: root)
    assert not process_tree.terminate_and_reap([owner], timeout=0.01)


@pytest.mark.parametrize("site", ["discover", "descendant_suspend", "descendant_kill"])
def test_descendant_faults_fail_closed_and_owner_is_still_reaped(monkeypatch, site):
    child = _FakePsProcess(pid=42, created=2.0)
    root = _FakePsProcess(children=[child])
    if site == "discover":
        root.children_error = psutil.AccessDenied(root.pid)
    elif site == "descendant_suspend":
        child.suspend_error = OSError("injected descendant suspend failure")
    else:
        child.kill_error = psutil.AccessDenied(child.pid)
    owner = _FakeOwner(root)
    replacement = _FakePsProcess(created=3.0)
    monkeypatch.setattr(
        process_tree.psutil,
        "Process",
        lambda pid: root if pid == root.pid and root.alive else replacement,
    )
    assert not process_tree.terminate_and_reap([owner])
    assert not owner.is_alive()
    if site == "descendant_suspend":
        assert not child.alive


def test_root_kill_error_fails_closed(monkeypatch):
    root = _FakePsProcess()
    owner = _FakeOwner(root)

    def denied_kill():
        owner.kill_count += 1
        raise OSError("injected owner kill failure")

    owner.kill = denied_kill
    monkeypatch.setattr(process_tree.psutil, "Process", lambda _pid: root)
    assert not process_tree.terminate_and_reap([owner])


def test_post_cleanup_identity_uncertainty_fails_closed(monkeypatch):
    root = _FakePsProcess()
    owner = _FakeOwner(root)

    def process_for_pid(_pid):
        if root.alive:
            return root
        raise psutil.AccessDenied(root.pid)

    monkeypatch.setattr(process_tree.psutil, "Process", process_for_pid)
    assert not process_tree.terminate_and_reap([owner])


def test_queue_close_failure_is_reported_without_blocking():
    class Queue:
        def cancel_join_thread(self):
            raise OSError("injected queue failure")

    assert not process_tree.close_process_queue(Queue())


def test_unstable_discovery_exhaustion_cleans_every_known_identity(monkeypatch):
    discovered = []
    root = _FakePsProcess()
    owner = _FakeOwner(root)
    replacement = _FakePsProcess(created=99.0)

    def ever_growing_tree(recursive):
        assert recursive
        child = _FakePsProcess(pid=100 + len(discovered), created=2.0)
        discovered.append(child)
        return [child]

    root.children = ever_growing_tree
    monkeypatch.setattr(
        process_tree.psutil,
        "Process",
        lambda pid: root if pid == root.pid and root.alive else replacement,
    )
    monkeypatch.setattr(
        process_tree.psutil,
        "wait_procs",
        lambda items, timeout: (list(items), []),
    )
    assert not process_tree.terminate_and_reap([owner], timeout=0.01)
    assert len(discovered) == 16
    assert not owner.is_alive()
    assert all(not child.alive for child in discovered)


def test_disappearing_descendant_is_a_narrow_success(monkeypatch):
    child = _FakePsProcess(pid=42, created=2.0)
    root = _FakePsProcess(children=[child])
    replacement = _FakePsProcess(created=3.0)
    owner = _FakeOwner(root)
    original_suspend = child.suspend

    def disappear_after_suspend():
        original_suspend()
        child.alive = False

    child.suspend = disappear_after_suspend
    monkeypatch.setattr(
        process_tree.psutil,
        "Process",
        lambda pid: root if pid == root.pid and root.alive else replacement,
    )
    assert process_tree.terminate_and_reap([owner])
    assert child.kill_count == 0
