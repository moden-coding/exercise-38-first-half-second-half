#!/usr/bin/env python3
"""Tests for the First Half Second Half assignment."""

import unittest
from unittest.mock import patch

import numpy as np

from src.first_half_second_half import first_half_second_half


class TestFirstHalfSecondHalf(unittest.TestCase):
    """first_half_second_half(a) -> rows where sum(first half) > sum(second half)."""

    def test_shape(self):
        n = 10
        for m in range(2, 8):
            a = np.random.randn(n, 2 * m)
            result = first_half_second_half(a)
            self.assertEqual(
                result.shape[1],
                2 * m,
                msg="Incorrect number of columns for array %s: expected "
                "%d, got %d." % (a, 2 * m, result.shape[1]),
            )
            self.assertLessEqual(
                result.shape[0],
                n,
                msg="There cannot be more rows (%d) than in the input "
                "array %s (%d rows)." % (result.shape[0], a, n),
            )

    def test_simple(self):
        n = 10
        a = np.random.randn(n, 2)
        result = first_half_second_half(a)
        correct = np.sum(a[:, 0] > a[:, 1])
        self.assertEqual(
            result.shape[1],
            2,
            msg="Incorrect number of columns for random array %s" % a,
        )
        self.assertEqual(
            result.shape[0],
            correct,
            msg="Wrong result for random array %s: expected %d matching "
            "rows, got %d." % (a, correct, result.shape[0]),
        )

    def test_content(self):
        n = 10
        for m in range(2, 8):
            a = np.random.randn(n, 2 * m)
            result = first_half_second_half(a)
            for row in result:
                self.assertGreater(
                    np.sum(row[0:m]),
                    np.sum(row[m:]),
                    msg="Wrong result for array %s: row %s should have "
                    "sum(first half) > sum(second half)." % (a, row),
                )

    def test_calls(self):
        n = 10
        m = 4
        a = np.random.randn(n, 2 * m)
        with patch("numpy.sum", side_effect=np.sum) as psum:
            first_half_second_half(a)
            self.assertEqual(
                psum.call_count,
                2,
                msg="Expected exactly two calls to np.sum (once per half "
                "of the array), got %d." % psum.call_count,
            )


if __name__ == "__main__":
    unittest.main()
