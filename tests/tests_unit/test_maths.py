"""
Tests for the maths module
"""
import pytest
import numpy as np

from practical_pytest import maths


def test_mean():
    """
    Check if we can find the mean of some integers
    """
    numbers = [1, 2, 3]

    expected_mean = 2
    calculated_mean = maths.mean(numbers)

    assert expected_mean == calculated_mean


def test_mean_complex():
    """
    Check that we can find the mean of some complex numbers
    """
    numbers = [2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 1.0j]

    expected_mean = 3.0 + 2.0j
    calculated_mean = maths.mean(numbers)

    assert np.isclose(calculated_mean, expected_mean)


def test_mean_floats():
    """
    Check that we can find the mean of some floats
    """
    numbers = [1.0, 2.5, 3.5]

    expected_mean = 7 / 3
    calculated_mean = maths.mean(numbers)

    assert np.isclose(expected_mean, calculated_mean)


def test_mean_negative():
    """
    Check that we can find the mean of some negative floats
    """
    numbers = [-1.0, -5.0, -3.5]

    expected_mean = -9.5 / 3
    calculated_mean = maths.mean(numbers)

    assert np.isclose(expected_mean, calculated_mean)


def test_mean_empty():
    """
    Maybe for my purposes, I expect the mean of an empty list to evaluate to 0

    """
    assert maths.mean([]) == 0.0
