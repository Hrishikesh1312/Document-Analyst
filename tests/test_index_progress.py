from __future__ import annotations

import unittest

from document_analyst.services.index_progress import IndexProgressEvent, IndexProgressTracker


class IndexProgressTests(unittest.TestCase):
    def test_overall_progress_moves_with_units_and_across_files(self):
        tracker = IndexProgressTracker()
        fractions = []
        events = [IndexProgressEvent("hashing", "b.pdf", 2, 2, True)]
        for index in (1, 2):
            for phase in ("reading", "chunking", "embedding", "writing"):
                for done in (0, 50, 100):
                    events.append(IndexProgressEvent(phase, "book.pdf", index, 2,
                                                     done == 100, done, 100, "passages"))
        for timestamp, event in enumerate(events):
            tracker.observe(event, timestamp)
            fractions.append(tracker.overall_fraction)
        self.assertEqual(fractions, sorted(fractions))
        self.assertGreater(fractions[5], fractions[4])
        self.assertEqual(fractions[-1], 1.0)

    def test_estimate_waits_for_batches_excludes_load_and_resets_for_next_stage(self):
        tracker = IndexProgressTracker()
        for done, timestamp in ((0, 0), (32, 100), (32, 101), (64, 102)):
            tracker.observe(IndexProgressEvent("chunking", "book.pdf", 1, 1,
                                              False, done, 160, "passages"), timestamp)
        self.assertIsNone(tracker.throughput(102))
        tracker.observe(IndexProgressEvent("chunking", "book.pdf", 1, 1,
                                          False, 96, 160, "passages"), 104)
        self.assertAlmostEqual(tracker.throughput(104), 16.0)
        self.assertAlmostEqual(tracker.remaining_seconds(104), 4.0)
        self.assertLess(tracker.throughput(110), 16.0)
        tracker.observe(IndexProgressEvent("embedding", "book.pdf", 1, 1,
                                          False, 0, 50, "passages"), 111)
        self.assertIsNone(tracker.remaining_seconds(111))
        tracker.observe(IndexProgressEvent("embedding", "next.pdf", 2, 2,
                                          False, 32, 50, "passages"), 112)
        self.assertEqual(len(tracker.samples), 1)

    def test_empty_counts_and_skipped_files(self):
        self.assertEqual(IndexProgressEvent("reading", "empty.pdf", 0, 0).overall_fraction, 0)
        event = IndexProgressEvent("skipped", "unchanged.txt", 3, 3, True)
        self.assertEqual(event.overall_fraction, 1)
        self.assertEqual(IndexProgressEvent("reading", "book", 1, 1, False, 200, 100).stage_fraction, 1)
