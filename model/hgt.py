"""Heterogeneous Graph Transformer (HGT) model for JobGraph."""
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

from extraction.schema import EMBEDDING_DIM


class JobGraphHGT(nn.Module):
    """HGT model that produces node embeddings for the job knowledge graph.

    Architecture:
    - Linear projection per node type to hidden_channels
    - N layers of HGTConv (heterogeneous attention)
    - Output: EMBEDDING_DIM per node
    """

    def __init__(
        self,
        metadata: tuple,  # (node_types, edge_types) from HeteroData
        hidden_channels: int = EMBEDDING_DIM,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Input projection per node type
        self.input_lins = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })

        # HGT convolution layers
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Forward pass -- returns embeddings per node type.

        Args:
            x_dict: {node_type: tensor} input features
            edge_index_dict: {edge_type: tensor} edge indices

        Returns:
            {node_type: tensor} embeddings of shape (N, hidden_channels)
        """
        # Project to hidden dim
        x_dict = {
            node_type: self.input_lins[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        # HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

    def encode_jobs(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Convenience: forward pass + return only job embeddings."""
        out = self.forward(x_dict, edge_index_dict)
        return out["job"]
