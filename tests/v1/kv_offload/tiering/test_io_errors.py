# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""errno classification of tier I/O failures and its plumbing to
``JobResult.error_errno``."""

import errno
import os

import pytest

from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.io_errors import (
    IOFailureKind,
    classify_errno,
    indicates_tier_pressure,
)

from .test_fs_tier import (
    _BLOCK_ELEMENTS,
    _MOCK_OFFLOADING_SPEC,
    _page_aligned_zero_tensor,
    drain,
    key,
    make_job,
)


@pytest.mark.parametrize(
    "err, kind",
    [
        (None, IOFailureKind.UNKNOWN),
        (errno.ENOSPC, IOFailureKind.CAPACITY),
        (errno.EDQUOT, IOFailureKind.CAPACITY),
        (errno.EFBIG, IOFailureKind.CAPACITY),
        (errno.EINTR, IOFailureKind.TRANSIENT),
        (errno.EAGAIN, IOFailureKind.TRANSIENT),
        (errno.EMFILE, IOFailureKind.TRANSIENT),
        (errno.ENFILE, IOFailureKind.TRANSIENT),
        (errno.ENOMEM, IOFailureKind.TRANSIENT),
        (errno.EIO, IOFailureKind.PERSISTENT),
        (errno.EACCES, IOFailureKind.PERSISTENT),
        (errno.EROFS, IOFailureKind.PERSISTENT),
    ],
)
def test_classify_errno(err, kind):
    assert classify_errno(err) is kind


def test_only_transient_failures_skip_back_pressure():
    assert [k for k in IOFailureKind if not indicates_tier_pressure(k)] == [
        IOFailureKind.TRANSIENT
    ]


def _raise(exc):
    def task():
        raise exc

    return task


def _wait_finished(pool, n):
    pool.wait_idle()
    finished = pool.get_finished()
    assert len(finished) == n
    return {job_id: rest for job_id, *rest in finished}


def test_thread_pool_reports_first_failing_errno():
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    try:
        pool.enqueue_store(1, 1, [lambda: None])
        pool.enqueue_store(2, 1, [_raise(OSError(errno.ENOSPC, "full"))])
        pool.enqueue_store(3, 1, [_raise(ValueError("not an OSError"))])
        pool.enqueue_load(4, 1, [_raise(OSError("no errno"))])
        results = _wait_finished(pool, 4)
    finally:
        pool.shutdown()

    assert results[1][0] is True and results[1][2] is None
    assert results[2][0] is False and results[2][2] == errno.ENOSPC
    assert results[3][0] is False and results[3][2] is None
    assert results[4][0] is False and results[4][2] is None


def test_thread_pool_keeps_errno_of_first_failed_task():
    # One thread runs the tasks in order, so EDQUOT fails first.
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    try:
        pool.enqueue_store(
            7,
            3,
            [
                lambda: None,
                _raise(OSError(errno.EDQUOT, "quota")),
                _raise(OSError(errno.EIO, "io")),
            ],
        )
        results = _wait_finished(pool, 1)
    finally:
        pool.shutdown()

    assert results[7][0] is False
    assert results[7][2] == errno.EDQUOT


def _tier(tmp_path, num_blocks=2):
    tensor = _page_aligned_zero_tensor(num_blocks, _BLOCK_ELEMENTS)
    return FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
    )


def _job(job_id, n, is_promotion=False):
    keys = [key(100 * job_id + i) for i in range(n)]
    return make_job(job_id, keys, list(range(n)), is_promotion=is_promotion)


def test_fs_tier_reports_store_errno(tmp_path):
    tier = _tier(tmp_path)
    job = _job(1, 1)
    # A regular file where the block's directory should be makes the store fail.
    block_dir = os.path.dirname(tier.file_mapper.get_file_name(next(iter(job.keys))))
    os.makedirs(os.path.dirname(block_dir), exist_ok=True)
    open(block_dir, "w").close()
    try:
        tier.submit_store(job)
        (result,) = drain(tier)
    finally:
        tier.shutdown()

    assert result.success is False
    assert result.error_errno == errno.ENOTDIR


def test_fs_tier_reports_load_errno_and_none_on_success(tmp_path):
    tier = _tier(tmp_path)
    try:
        tier.submit_store(_job(1, 1))
        (stored,) = drain(tier)
        # Nothing was stored under job 2's keys, so the load cannot open them.
        tier.submit_load(_job(2, 1, is_promotion=True))
        (loaded,) = drain(tier)
    finally:
        tier.shutdown()

    assert stored.success is True and stored.error_errno is None
    assert loaded.success is False and loaded.error_errno == errno.ENOENT
