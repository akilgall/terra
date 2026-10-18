"""Opt-in: re-tokenizing CosMx NSCLC with the TERRA-96M bundle reproduces a
dataset tokenized by the base code byte-for-byte.

Set ``TERRA_GOLDEN_DATA=1``, ``TERRA_GOLDEN_ADATA`` (harmonized ``.h5ad``),
``TERRA_GOLDEN_DATASET`` (``save_to_disk`` folder tokenized by the base code)
and ``TERRA_BUNDLE_DIR`` (TERRA-96M bundle).
"""

import os

import numpy as np
import pytest

REQUIRED = ("TERRA_GOLDEN_DATA", "TERRA_GOLDEN_ADATA", "TERRA_GOLDEN_DATASET",
            "TERRA_BUNDLE_DIR")

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in REQUIRED),
    reason="set " + ", ".join(REQUIRED) + " to run the CosMx tokenizer golden check")


def test_cosmx_retokenization_is_byte_identical(tmp_path):
    ad = pytest.importorskip("anndata")
    datasets = pytest.importorskip("datasets")
    from terra.inference.tokenize import tokenize_adata

    adata = ad.read_h5ad(os.environ["TERRA_GOLDEN_ADATA"])
    # Compare raw stored values (a saved torch format would round to float32).
    expected = datasets.load_from_disk(os.environ["TERRA_GOLDEN_DATASET"]).with_format(None)

    actual = tokenize_adata(adata, os.environ["TERRA_BUNDLE_DIR"], str(tmp_path), nproc=1,
                            processing_mode="sequential", use_generator=False)

    assert len(actual) == len(expected)
    actual = actual.with_format(None)
    for i in range(len(expected)):
        got, want = actual[i], expected[i]
        assert got["cell_id"] == want["cell_id"]
        for key in ("gene_tokens", "gene_expr", "rel_x_coord", "rel_y_coord"):
            assert np.array_equal(np.asarray(got[key]), np.asarray(want[key])), (i, key)
        assert got["n_nonzero_tokens"] == want["n_nonzero_tokens"]
