import pytest

from cua.models.base import ModelAdapter


def test_model_adapter_is_abstract():
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]
