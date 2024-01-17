import torch
import torch.nn as nn
import numpy as np


def build_square_matrix(size):
    matrix = np.zeros((size, size), dtype=float)

    # Fill the main diagonal with ones
    np.fill_diagonal(matrix, 1.0)

    # Fill the adjacent diagonals with 0.5
    np.fill_diagonal(matrix[:, 1:], 0.5)
    np.fill_diagonal(matrix[1:, :], 0.5)

    # Fill other positions with values proportional to the distance from the diagonal
    for i in range(size):
        for j in range(size):
            if matrix[i, j] == 0.0:
                distance_from_diagonal = abs(i - j)
                matrix[i, j] = 1.0 / distance_from_diagonal

    return torch.tensor(matrix)


class GraphAttentionLayer(nn.Module):
    """
        Graph Attention Layer. Inspired by https://arxiv.org/abs/1710.10903

        Args:
            n_nodes (int): Number of nodes in the graph.
            in_features (int): Number of features per node.
            out_features (int): Number of output features per node.
            dropout_prob (float): Dropout probability for node dropout.
            alpha (float): Negative slope used in the LeakyReLU activation function.
            is_training (bool): Flag indicating whether the model is in training mode.

        Attributes:
            W (nn.Parameter): Learnable weight matrix for linear transformation.
            a (nn.Parameter): Learnable attention parameter.
    """

    def __init__(self, n_nodes, in_features, out_features, dropout_prob, alpha, is_training: bool = True):
        super(GraphAttentionLayer, self).__init__()

        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.is_training = is_training

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
            Forward pass for the Graph Attention Layer.

            Args:
                h (torch.Tensor): Node features of shape (n_batch, n_nodes, n_features).
                adj (torch.Tensor): Adjacency matrix of shape (n_nodes, n_nodes).

            Returns:
                torch.Tensor: New node embeddings of shape (n_batch, n_nodes, out_features).
        """

        hW = torch.matmul(h, self.W)  # xW shape -> (n_batch, n_nodes, out_features)

        hW_1 = torch.matmul(hW, self.a[:self.out_features, :])  # shape -> (n_batch, n_nodes, 1)
        hW_2 = torch.matmul(hW, self.a[self.out_features:, :])  # shape -> (n_batch, n_nodes, 1)
        e = hW_1 + torch.transpose(input=hW_2, dim0=2, dim1=1)  # (broadcast add) -> (n_batch, n_nodes, n_nodes)
        e = self.leakyrelu(e)
        e = torch.softmax(e, dim=2)
        e = torch.dropout(input=e, p=self.dropout_prob, train=self.is_training)

        zero_vec = -9e15 * torch.ones_like(e)  # shape -> (n_batch, n_nodes, n_nodes)
        adj = adj.to(e.device)
        attention = torch.where(adj > 0, e, zero_vec)  # shape -> (n_batch, n_nodes, n_nodes)

        new_h = self.sigmoid(torch.matmul(attention, hW))  # shape -> (n_batch, n_nodes, out_features)
        return new_h


class MultiHeadGraphAttentionLayer(nn.Module):
    """
        Multi-Head Graph Attention Layer.

        Args:
            n_heads (int): Number of attention heads.
            n_nodes (int): Number of nodes in the graph.
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            dropout_prob (float): Dropout probability for node dropout.
            alpha (float): Negative slope used in the LeakyReLU activation function.
            is_training (bool): Flag indicating whether the model is in training mode.

        Attributes:
            graph_att (GraphAttentionLayer): Single head Graph Attention Layer.
            heads_list (nn.ModuleList): List of Graph Attention Layers.
            avg (nn.Linear): Linear layer for aggregation of the different head-results
        """
    def __init__(self, n_heads, n_nodes, in_features, out_features, dropout_prob: float = 0.5, alpha: float = 0.2,
                 is_training: bool = True):
        assert n_heads > 0

        super(MultiHeadGraphAttentionLayer, self).__init__()

        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features

        self.graph_att = GraphAttentionLayer(n_nodes=n_nodes,
                                             in_features=in_features,
                                             out_features=out_features,
                                             dropout_prob=dropout_prob,
                                             alpha=alpha,
                                             is_training=is_training)

        self.n_heads = n_heads
        self.heads_list = nn.ModuleList()
        for _ in range(n_heads):
            self.heads_list.append(GraphAttentionLayer(n_nodes=n_nodes,
                                                       in_features=in_features,
                                                       out_features=out_features,
                                                       dropout_prob=dropout_prob,
                                                       alpha=alpha,
                                                       is_training=is_training))

        self.avg = nn.Linear(in_features=n_heads, out_features=1)

    def forward(self, x, adj):
        """
        Forward pass for the Multi-Head Graph Attention Layer.

        Args:
            x (torch.Tensor): Node features of shape (n_batch, n_nodes, d_embed).
            adj (torch.Tensor): Adjacency matrix of shape (n_nodes, n_nodes).

                Returns:
                    torch.Tensor: New node embeddings of shape (n_batch, n_nodes, d_embed).
                """
        n_batch, n_nodes, d_embed = x.shape
        out_heads = torch.empty(size=(n_batch, n_nodes, d_embed, self.n_heads)).to(x.device)

        for idx in range(self.n_heads):
            out_heads[:, :, :, idx] = self.heads_list[idx](x, adj)

        out = self.avg(out_heads).squeeze(dim=-1)  # (n_batch, n_nodes, d_embed)

        return out