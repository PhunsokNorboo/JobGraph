"""Evaluation metrics for link prediction."""
import torch
import numpy as np


def evaluate_link_prediction(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k_values: list[int] = [1, 5, 10],
) -> dict:
    """Evaluate link prediction quality.

    Args:
        predictions: model scores (logits)
        labels: binary labels (1=positive, 0=negative)
        k_values: values of k for Hits@k

    Returns:
        Dict with mrr, hits_at_1, hits_at_5, hits_at_10, ndcg_at_10
    """
    pred = predictions.cpu().numpy()
    lab = labels.cpu().numpy()

    # Separate positive and negative
    pos_mask = lab == 1
    neg_mask = lab == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0, "ndcg_at_10": 0.0}

    pos_scores = pred[pos_mask]
    neg_scores = pred[neg_mask]

    # For each positive, compute rank among negatives
    ranks = []
    for p_score in pos_scores:
        rank = (neg_scores >= p_score).sum() + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=float)

    metrics = {}

    # MRR
    metrics["mrr"] = float(np.mean(1.0 / ranks))

    # Hits@K
    for k in k_values:
        metrics[f"hits_at_{k}"] = float(np.mean(ranks <= k))

    # NDCG@10
    k = 10
    dcg = np.sum(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0))
    ideal_dcg = np.sum(1.0 / np.log2(np.arange(1, min(len(ranks), k) + 1) + 1))
    metrics["ndcg_at_10"] = float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    return metrics
