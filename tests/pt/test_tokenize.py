import pytest
from src.zachakit.pt.tokenize import UnitDS


@pytest.fixture
def unitds():
    return UnitDS(
        cache_dir="draft/cache",
        from_local_file=True,
        text_col="Content",
        root="draft/zh_cn.parquet",
        data_type="parquet",
        block_size=1024,
        tokenizer_name="hfl/chinese-alpaca-2-1.3b",
    )


def test_unitds_root_file(unitds):
    assert unitds.root == "draft/zh_cn.parquet"


def test_unitds_living_check(unitds):
    unitds.living_check()
    assert unitds._alive


def test_unitds_tokenize_group(unitds):
    unitds.living_check()
    assert unitds.tokenize_group()
