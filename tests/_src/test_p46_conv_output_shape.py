import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e


from p46_conv_output_shape import *

def test_p46_conv_output_shape():

    with step("3x3 conv with stride 1, padding 1 preserves size"):
        assert conv_out_shape(32, kernel=3, stride=1, padding=1) == 32
    with step("5x5 conv with stride 2, padding 2 halves size"):
        assert conv_out_shape(32, kernel=5, stride=2, padding=2) == 16
    with step("3x3 stride-2 no-padding shrinks 7 -> 3"):
        assert conv_out_shape(7, kernel=3, stride=2, padding=0) == 3
    with step("dilation=2 effective kernel: 7 -> 2"):
        assert conv_out_shape(7, kernel=3, stride=2, padding=0, dilation=2) == 2
    with step("chain: classic ImageNet stem 224 -> 112 -> 56 -> 56"):
        layers = [
            {"kernel": 7, "stride": 2, "padding": 3},
            {"kernel": 3, "stride": 2, "padding": 1},
            {"kernel": 3, "stride": 1, "padding": 1},
        ]
        assert conv_chain_shape(224, layers) == 56
    with step("transposed conv: 7 -> 14 with k=4 s=2 p=1"):
        assert transposed_conv_out_shape(7, kernel=4, stride=2, padding=1) == 14
    with step("transposed conv: 28 -> 56 with k=2 s=2 p=0"):
        assert transposed_conv_out_shape(28, kernel=2, stride=2, padding=0) == 56

