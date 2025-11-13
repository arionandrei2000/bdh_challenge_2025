import torch
import torch.nn as nn

class MiniBulkRNABert(nn.Module):
    def __init__(self, num_genes, num_bins=64, d_model=128, num_layers=2, n_heads=4):
        super().__init__()

        # Expression embedding (token embedding)
        self.token_embed = nn.Embedding(num_bins, d_model)

        # Gene embedding
        self.gene_embed = nn.Embedding(num_genes, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM prediction head
        self.mlm_head = nn.Linear(d_model, num_bins)

    def forward(self, token_ids, gene_ids):
        """
        token_ids: (batch, num_genes)
        gene_ids: (num_genes,) constant index tensor 0..num_genes-1
        """
        x = self.token_embed(token_ids) + self.gene_embed(gene_ids)
        h = self.encoder(x)
        logits = self.mlm_head(h)
        return logits
