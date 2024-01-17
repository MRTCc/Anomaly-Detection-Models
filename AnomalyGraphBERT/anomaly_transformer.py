import os
import sys
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from transformer import get_transformer_encoder


# Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, embedding, transformer_encoder, mlp_layers, d_embed, patch_size, max_seq_len):
        """
            Anomaly Transformer model for sequence data anomaly detection. Paper: https://arxiv.org/abs/2305.04468v1

            Args:
                embedding (nn.Module): Embedding layer to feed data into the Transformer encoder.
                transformer_encoder (nn.Module): Transformer encoder body.
                mlp_layers (nn.Module): MLP layers to return output data.
                d_embed (int): Embedding dimension in the Transformer encoder.
                patch_size (int): Number of data points for an embedded vector.
                max_seq_len (int): Maximum length of the sequence (window size).
        """

        super(AnomalyTransformer, self).__init__()
        self.embedding = embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len

    def forward(self, x):
        """
            Forward pass for the Anomaly Transformer model.

            Args:
                x (torch.Tensor): Input tensor of shape (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _).

            Returns:
                torch.Tensor: Output tensor of shape (n_batch, data_seq_len, d_embed).
        """
        n_batch, n_samples = x.shape[0], x.shape[1]

        # embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        embedded_out = x.view(n_batch, int(n_samples / self.patch_size), -1)
        embedded_out = self.embedding(embedded_out)

        transformer_out = self.transformer_encoder(embedded_out)  # Encode data.
        output = self.mlp_layers(transformer_out)  # Reconstruct data.

        return output.view(n_batch, n_samples, -1).squeeze(-1)

        # questa è per compatibilità con il train.py originale
        # return output.view(n_batch, n_samples, -1)


class TransformerEmbedder(nn.Module):
    """
    Transformer-based feature embedding for Anomaly Transformer.

    This module performs feature embedding using a Transformer encoder with one layer and a single attention head,
    followed by a linear layer.
    It is used in the Anomaly Transformer model to transform the input data into a higher-dimensional feature space.

    Args:
        in_features (int): The dimension of the input features.
        seq_len (int): The total sequence length (window size).
        max_seq_len (int): The maximum sequence length.
        out_features (int): The dimension of the output features.
        dropout_prob (float): Dropout probability for the Transformer encoder.

    Attributes:
        transformer_encoder (nn.Module): Transformer encoder with one layer and one attention head.
        linear (nn.Linear): Linear layer to map the embedded features to the desired output dimension.

    Example:
        embedder = TransformerEmbedder(in_features=64, seq_len=512, max_seq_len=512, out_features=256)
        embedded_features = embedder(input_data)
    """
    def __init__(self, in_features, seq_len, max_seq_len, out_features, dropout_prob=0.1):
        super(TransformerEmbedder, self).__init__()

        # transformer encoder di un layer con una sola head
        self.transformer_encoder = get_transformer_encoder(d_embed=in_features,
                                                           positional_encoding=None,
                                                           n_layer=1,
                                                           max_seq_len=max_seq_len,
                                                           dropout=dropout_prob)
        # linear layer
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        """
        Forward pass of the TransformerEmbedder.

        Args:
            x (torch.Tensor): Input tensor of shape (n_batch, n_patch, n_features_per_patch).

        Returns:
            torch.Tensor: Output tensor of shape (n_batch, n_patch, out_features).
        """
        return self.linear(self.transformer_encoder(x))


# Get Anomaly Transformer.
def get_anomaly_transformer(input_d_data,
                            output_d_data,
                            patch_size,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            positional_encoding=None,
                            transformer_n_layer=6,
                            dropout=0.1,
                            n_heads=3,
                            alpha=0.1,
                            is_training=False):
    """
    Factory function to create an Anomaly Transformer model.

    Args:
        input_d_data (int): Data input dimension.
        output_d_data (int): Data output dimension.
        patch_size (int): Number of data points per embedded feature.
        d_embed (int): Embedding dimension (in Transformer encoder).
        hidden_dim_rate (float): Hidden layer dimension rate to d_embed.
        max_seq_len (int): Maximum length of the sequence (window size).
        positional_encoding (str or None): Positional encoding for embedded input; None/Sinusoidal/Absolute.
        transformer_n_layer (int): Number of Transformer encoder layers.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads.
        n_nodes (int): Number of nodes in the graph.
        alpha (float): Negative slope used in the LeakyReLU activation function.
        is_training (bool): Flag indicating whether the model is in training mode.

    Returns:
        AnomalyTransformer: An instance of the AnomalyTransformer model.
    """

    hidden_dim = int(hidden_dim_rate * d_embed)

    embedding = nn.Linear(in_features=input_d_data*patch_size, out_features=d_embed)

    n_nodes = int(max_seq_len / patch_size)
    transformer_encoder = get_transformer_encoder(n_nodes=n_nodes,
                                                  in_features=d_embed,
                                                  out_features=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  n_layer=transformer_n_layer,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout,
                                                  n_heads=n_heads,
                                                  alpha=alpha,
                                                  is_training=is_training)

    mlp_layers = nn.Sequential(nn.Linear(d_embed, hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim, output_d_data * patch_size))

    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)

    return AnomalyTransformer(embedding,
                              transformer_encoder,
                              mlp_layers,
                              d_embed,
                              patch_size,
                              max_seq_len)
