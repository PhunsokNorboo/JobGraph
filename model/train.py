"""Training loop for JobGraph HGT model."""
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

from model.hgt import JobGraphHGT
from model.evaluate import evaluate_link_prediction

logger = logging.getLogger(__name__)


def train(
    graph_path: Path,
    output_dir: Path,
    hidden_channels: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    lr: float = 0.001,
    epochs: int = 200,
    batch_size: int = 1024,
    eval_every: int = 10,
    device: str = "auto",
):
    """Train HGT model on link prediction task (job <-> skill).

    Args:
        graph_path: Path to hetero_data.pt
        output_dir: Where to save model checkpoint
        hidden_channels: Hidden dimension size
        num_heads: Number of attention heads per HGT layer
        num_layers: Number of HGT convolution layers
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for data loading
        eval_every: Evaluate on validation set every N epochs
        device: Device string ("auto", "cuda", "mps", "cpu")
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Training on device: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    data = torch.load(graph_path, weights_only=False)
    logger.info(f"Graph loaded: {data}")

    # Split edges for train/val/test
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[("job", "requires", "skill")],
        rev_edge_types=[("skill", "rev_requires", "job")],
    )
    train_data, val_data, test_data = transform(data)

    # Model
    model = JobGraphHGT(
        metadata=train_data.metadata(),
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mrr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        # Move data to device
        td = train_data.to(device)

        # Forward pass
        z_dict = model(td.x_dict, td.edge_index_dict)

        # Get positive and negative edges for link prediction
        edge_label_index = td["job", "requires", "skill"].edge_label_index
        edge_label = td["job", "requires", "skill"].edge_label

        # Compute predictions via dot product
        z_job = z_dict["job"]
        z_skill = z_dict["skill"]

        src = z_job[edge_label_index[0]]
        dst = z_skill[edge_label_index[1]]
        pred = (src * dst).sum(dim=-1)

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, edge_label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0

        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vd = val_data.to(device)
                vz = model(vd.x_dict, vd.edge_index_dict)

                val_edge_label_index = vd["job", "requires", "skill"].edge_label_index
                val_edge_label = vd["job", "requires", "skill"].edge_label

                val_src = vz["job"][val_edge_label_index[0]]
                val_dst = vz["skill"][val_edge_label_index[1]]
                val_pred = (val_src * val_dst).sum(dim=-1)
                val_loss = F.binary_cross_entropy_with_logits(val_pred, val_edge_label.float())

                metrics = evaluate_link_prediction(val_pred, val_edge_label)

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
                f"MRR: {metrics['mrr']:.4f} | Hits@10: {metrics['hits_at_10']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "metadata": train_data.metadata(),
                    "hidden_channels": hidden_channels,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "epoch": epoch,
                    "best_mrr": best_mrr,
                }, output_dir / "best_model.pt")
                logger.info(f"  -> New best model saved (MRR: {best_mrr:.4f})")

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_d = test_data.to(device)
        tz = model(test_d.x_dict, test_d.edge_index_dict)
        test_ei = test_d["job", "requires", "skill"].edge_label_index
        test_el = test_d["job", "requires", "skill"].edge_label
        test_src = tz["job"][test_ei[0]]
        test_dst = tz["skill"][test_ei[1]]
        test_pred = (test_src * test_dst).sum(dim=-1)
        test_metrics = evaluate_link_prediction(test_pred, test_el)

    logger.info("=== Test Results ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return test_metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parent.parent
    train(
        graph_path=Path(sys.argv[1]) if len(sys.argv) > 1 else base / "data" / "graph" / "hetero_data.pt",
        output_dir=base / "data" / "graph",
    )
