import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from q38_launch import sha256_file, verify_segments


class SegmentVerificationTest(unittest.TestCase):
    def fixture(self, root):
        paths = []
        for i in range(4):
            path = root / f'{i}.bin'
            path.write_bytes(bytes([i]) * 32)
            paths.append(path)
        alias = root / 'alias.bin'
        os.link(paths[0], alias)
        paths.append(alias)
        index = root / 'index.q38d'
        index.write_text('Q38_DEVICE_INDEX_V1\n' + ''.join(
            f'segment\t{p.name}\t32\t{sha256_file(p)}\n' for p in paths))
        return index, paths

    def test_parallel_hashes_are_bounded_and_hardlinks_hashed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _ = self.fixture(Path(directory))
            expected = verify_segments([index], True)
            lock, barrier = threading.Lock(), threading.Barrier(2, timeout=5)
            active = peak = calls = 0
            def observed(path):
                nonlocal active, peak, calls
                with lock:
                    active += 1
                    peak = max(active, peak)
                    calls += 1
                try:
                    barrier.wait()
                    return sha256_file(path)
                finally:
                    with lock:
                        active -= 1
            with patch('q38_launch.sha256_file', side_effect=observed):
                self.assertEqual(verify_segments([index], True, 2), expected)
            self.assertEqual(expected, {'segments':4, 'segment_bytes':128})
            self.assertEqual((peak, calls, active), (2, 4, 0))

    def test_corruption_fails_for_serial_and_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            index, paths = self.fixture(Path(directory))
            paths[2].write_bytes(b'x' * 32)
            for workers in (1, 4):
                with self.assertRaisesRegex(ValueError, 'digest differs'):
                    verify_segments([index], True, workers)

    def test_conflicting_hardlink_digest_fails_even_without_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _ = self.fixture(Path(directory))
            with index.open('a') as output:
                output.write('segment\talias.bin\t32\t' + '0' * 64 + '\n')
            with self.assertRaisesRegex(ValueError, 'conflicting hard-link'):
                verify_segments([index], False, 4)

    def test_mutation_during_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            index, _ = self.fixture(Path(directory))
            def mutate(path):
                digest = sha256_file(path)
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
                return digest
            with patch('q38_launch.sha256_file', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed during verification'):
                    verify_segments([index], True, 2)

    def test_invalid_parallelism_is_rejected(self):
        for workers in (0, 17):
            with self.assertRaises(ValueError):
                verify_segments([], True, workers)


if __name__ == '__main__':
    unittest.main()
