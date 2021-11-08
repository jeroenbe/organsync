from packaging import version

import organsync


def test_sanity() -> None:
    assert version.parse(organsync.version.__version__) is not None
