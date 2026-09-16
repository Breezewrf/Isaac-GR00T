from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from analyze_action_smoothness import classify_dimension  # noqa: E402


def _classify(sigma: float, max_abs_delta: float) -> str:
    return classify_dimension(
        sigma,
        max_abs_delta,
        smooth_threshold=0.005,
        jerky_threshold=0.01,
        inactive_epsilon=1e-8,
    )


def test_classify_dimension():
    assert _classify(0.0, 0.0) == "inactive/discrete"
    assert _classify(0.0049, 0.2) == "smooth"
    assert _classify(0.005, 0.2) == "moderate"
    assert _classify(0.0099, 0.2) == "moderate"
    assert _classify(0.01, 0.2) == "jerky"
