"""Numerical checks for the statistics used by runtime dataset mixtures."""

from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
import numpy as np
import pytest


def _stats(values):
    return {
        "joint": {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "q01": np.quantile(values, 0.01, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
        }
    }


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_count_weights_match_concatenated_population_statistics(relative, offset):
    a = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 8.0]]) + offset
    b = np.array([[10.0, 20.0], [12.0, 24.0]]) + offset
    if relative:
        a, b = a[:, None, :], b[:, None, :]
    sources = [_stats(a), _stats(b)]
    result = merge_statistics(sources, [len(a), len(b)], is_relative_stats=relative)["joint"]
    pooled = np.concatenate([a, b])
    np.testing.assert_allclose(result["mean"], pooled.mean(axis=0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        result["std"], pooled.std(axis=0), rtol=1e-12, atol=np.spacing(offset)
    )
    np.testing.assert_array_equal(result["min"], pooled.min(axis=0))
    np.testing.assert_array_equal(result["max"], pooled.max(axis=0))
    np.testing.assert_array_equal(
        result["q01"], np.minimum(sources[0]["joint"]["q01"], sources[1]["joint"]["q01"])
    )
    np.testing.assert_array_equal(
        result["q99"], np.maximum(sources[0]["joint"]["q99"], sources[1]["joint"]["q99"])
    )


@pytest.mark.parametrize("relative", [False, True])
def test_custom_weights_preserve_small_variance_at_large_mean(relative):
    shape = (3, 2) if relative else (2,)
    sources = []
    for mean, std in [(1e9, 1.0), (1e9 + 2, 2.0)]:
        sources.append(
            {
                "joint": {
                    key: np.full(shape, value)
                    for key, value in {
                        "mean": mean,
                        "std": std,
                        "min": mean - 2,
                        "max": mean + 2,
                        "q01": mean - 1,
                        "q99": mean + 1,
                    }.items()
                }
            }
        )
    result = merge_statistics(sources, [3, 1], is_relative_stats=relative)["joint"]
    np.testing.assert_array_equal(result["mean"], np.full(shape, 1e9 + 0.5))
    # .75 * (1 + .5**2) + .25 * (4 + 1.5**2) = 2.5.
    np.testing.assert_allclose(result["std"], np.full(shape, np.sqrt(2.5)), rtol=1e-15)
