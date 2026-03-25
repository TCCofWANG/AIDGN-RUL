import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Degradation-Timed Bases (DTB) module ---
class DegradationTimedBases(nn.Module):
    """
    Given age: [B, T, 1] (normalized cycle index),
    produce age-conditioned basis mixture: out -> [B, T, d_model]
    where out[t] = sum_k coeffs[t,k] * basis_k.
    """
    def __init__(self, seq_len, basis_dim, d_model, hidden=32):
        super().__init__()
        self.seq_len = seq_len
        self.basis_dim = basis_dim
        self.d_model = d_model

        # learnable global bases (basis_dim x d_model)
        self.basis = nn.Parameter(torch.randn(basis_dim, d_model) * 0.1)

        # small projector from scalar age -> basis logits
        self.coeff_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, basis_dim)  # produce logits for basis mixture
        )

    def forward(self, age):
        """
        age: [B, T, 1]
        returns: [B, T, d_model]
        """
        B, T, _ = age.shape
        coeff_logits = self.coeff_mlp(age.view(B * T, 1)).view(B, T, self.basis_dim)  # [B,T,K]
        coeffs = F.softmax(coeff_logits, dim=-1)  # [B,T,K]
        # expand bases: [1, K, d] -> [B, K, d]
        bases = self.basis.unsqueeze(0).expand(B, -1, -1)
        # out: [B, T, d] = coeffs [B,T,K] @ bases [B,K,d]
        # out = torch.bmm(coeffs, bases)  # [B, T, d_model]
        return bases


# --- Attention Pooling (linear scoring, no tanh) ---
class AttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, d_model]
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        pooled = (x * weights).sum(dim=1)  # [B, d_model]
        return pooled


# --- CLIP-style model with DTB integrated into attention ---
class CLIP(nn.Module):
    def __init__(self, configs, input_feature):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.input_feature = input_feature
        d_model = configs.d_model
        self.n_dtb = input_feature

        # Embedding (reuse your DataEmbedding_inverted)
        from layers.Embed import DataEmbedding_inverted
        self.enc_embedding = DataEmbedding_inverted(
            configs.input_length, d_model, configs.dropout
        )

        # old projectors kept for compatibility (not used as sole DTB)
        self.project = nn.Linear(1, input_feature)
        self.project_d_model = nn.Linear(self.input_length, d_model)

        # Degradation-Timed Bases module
        self.dtb = DegradationTimedBases(seq_len=self.input_length, basis_dim=self.n_dtb, d_model=d_model)

        # Attention projections (for cross fusion)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # scale for age bias contribution (learnable)
        self.age_attn_scale = nn.Parameter(torch.tensor(1.0))

        # LayerNorm + dropout
        self.dropout = nn.Dropout(configs.dropout)
        self.norm_sensor = nn.LayerNorm(d_model)
        self.norm_fused = nn.LayerNorm(d_model)

        # Temporal pooling
        self.temporal_pool = AttentionPooling(d_model)

        # MLP / regression head
        self.fc = nn.Sequential(
            nn.Linear(d_model, configs.d_ff),
            nn.LeakyReLU(),
            nn.Linear(configs.d_ff, configs.d_ff),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )

        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_age_as_basis(self, age):
        """
        age: [B, T, 1]
        returns: dtb_out: [B, T, d_model]
        """
        return self.dtb(age)

    def cross_attention_with_dtb(self, sensor, age_basis):
        """
        sensor: [B, T, d]
        age_basis: [B, T, d]
        returns: fused: [B, T, d]
        """
        B, T, d = sensor.shape
        Q = self.query_proj(sensor)   # [B, T, d]
        K = self.key_proj(sensor)     # [B, T, d]
        V = self.value_proj(sensor)   # [B, T, d]

        # base attention scores from sensor self-similarity
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(d)  # [B, T, T]

        # age-guided bias: how queries align with DTB at each time
        # age_bias_scores = (Q @ age_basis^T) -> [B, T, T]
        age_bias_scores = torch.bmm(Q, age_basis.transpose(1, 2)) / np.sqrt(d)
        # scale the bias (learnable)
        scale = attn_scores + self.age_attn_scale
        attn_scores = age_bias_scores.transpose(1, 2) @ scale

        # normalize and compute attention-weighted values
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, T]
        out = torch.bmm(attn_weights, V) / np.sqrt(d)            # [B, T, d]
        # residual & norm
        # out = out + sensor
        out = self.norm_fused(out)
        return out

    def forward(self, x_enc, aging):
        """
        x_enc: [B, T, C]
        aging: [B, T, 1]  normalized cycle index per timestep
        """
        # sensor embedding
        sensor_features = self.enc_embedding(x_enc, None)   # [B, T, d_model]
        sensor_features = self.norm_sensor(self.dropout(sensor_features))

        # produce DTB (age-conditioned bases mixture)
        dtb_features = self.encode_age_as_basis(aging.to(sensor_features.device))  # [B, T, d_model]

        # cross-attention fusion with DTB bias
        fused = self.cross_attention_with_dtb(sensor_features, dtb_features)  # [B, T, d_model]

        # temporal pooling + regression
        pooled = self.temporal_pool(fused)  # [B, d_model]
        out = self.fc(pooled)               # [B, 1]
        rul = torch.abs(out)    # [B, 1]
        return None, rul


# --- Example usage snippet ---
if __name__ == "__main__":
    class Cfg: pass
    cfg = Cfg()
    cfg.input_length = 30
    cfg.d_model = 128
    cfg.dropout = 0.1
    cfg.d_ff = 256

    model = CLIP(cfg, input_feature=14)

    x = torch.randn(32, 30, 14)  # sensors
    age = torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).repeat(32, 1, 1)  # [B, T, 1]
    _, rul = model(x, age)
    print(rul.shape)  # [32]
