"""Model registry. Heavy imports (implicit, torch, sklearn) happen only when a model is built."""

from __future__ import annotations

from collections.abc import Sequence

from vgrec.models.base import Recommender

MODEL_NAMES = ("popularity", "als", "ncf", "content", "hybrid", "hybrid_pop")


def build_model(name: str, params: dict | None = None, *,
                item_text: Sequence[str] | None = None) -> Recommender:
    """Construct an unfitted model from a config grid point.

    Hybrids wrap other fitted models, so experiment.run_hybrid builds them instead.
    """
    params = dict(params or {})
    if name == "popularity":
        from vgrec.models.popularity import Popularity

        return Popularity(**params)
    if name == "als":
        from vgrec.models.als import ALS

        return ALS(**params)
    if name == "ncf":
        from vgrec.models.ncf import NeuMF

        if "mlp_layers" in params:
            params["mlp_layers"] = tuple(params["mlp_layers"])
        return NeuMF(**params)
    if name == "content":
        from vgrec.models.content import ContentTfidf

        if item_text is None:
            raise ValueError("content model needs item_text (see data.load_items)")
        return ContentTfidf(item_text, **params)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
