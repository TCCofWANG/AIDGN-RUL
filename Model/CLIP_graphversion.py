import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------
# Graph utilities
# ---------------------------
def build_graph(x):
    """
    Build adjacency matrix from feature similarity.
    x: (B, N, D) -> batch, nodes, features
    """
    sim = torch.matmul(x, x.transpose(-1, -2))  # (B, N, N)
    adj = torch.softmax(sim, dim=-1)            # normalized adjacency
    return adj


class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # Graph propagation: (B, N, N) @ (B, N, D) -> (B, N, D)
        h = torch.matmul(adj, x)
        return self.fc(h)


def consistency_loss(z1, z2):
    """Cross-view consistency loss."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return 1 - (z1 * z2).sum(dim=-1).mean()


# ---------------------------
# CLIP with Cross-View Graph Consistency
# ---------------------------
class CLIP(nn.Module):
    """
    CLIP with Cross-View Graph Consistency Learning
    Inspired by: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_feature):
        super(CLIP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.d_model = configs.d_model
        self.pred_len = 1
        self.input_feature = input_feature

        # Embedding layers
        self.enc_embedding = nn.Linear(self.input_length, configs.d_model)
        self.project_age = nn.Linear(1, input_feature)
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)

        # Graph layers
        self.gcn_sensor = GraphLayer(configs.d_model, configs.d_model)
        self.gcn_aging = GraphLayer(configs.d_model, configs.d_model)

        # Normalization
        self.norm = nn.LayerNorm(configs.d_model)

        # Prediction head
        self.linear = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

        self.decoder = nn.Sequential( nn.Linear( input_feature, configs.d_ff, bias=True ),
                                      nn.ReLU(),
                                      nn.Dropout(configs.dropout),
                                      nn.Linear( configs.d_ff, 1 ) )

        self.decoder2 = nn.Sequential( nn.Linear( input_feature, configs.d_ff, bias=True ),
                                              nn.ReLU(), nn.Linear( configs.d_ff, 1 ) )

    def encode_age(self, age):
        """
        Encode aging features: (B, T, 1) -> (B, T, d_model)
        """
        aging = age.transpose(1, 2).to(self.device)  # (B, 1, T)
        aging = self.project_age(aging.permute(0, 2, 1))               # (B, T, input_feature)
        aging = self.project_d_model(aging.permute(0, 2, 1))           # (B, d_model, T)
        return aging                                  # (B, T, d_model)

    def forward(self, x_enc, aging, return_loss=False, lambda_cvgcl=0.1):
        """
        x_enc:   (B, T, F)  sensor data
        aging:   (B, T, 1)  aging index
        """
        # Encode
        sensor_features = self.enc_embedding(x_enc.permute(0, 2, 1))    # (B, T, d_model)
        aging_features = self.encode_age(aging)        # (B, T, d_model)

        # Normalize
        sensor_features = self.norm(sensor_features)
        aging_features = self.norm(aging_features)

        # Build graphs
        adj_sensor = build_graph(sensor_features)
        adj_aging = build_graph(aging_features)

        # Graph propagation
        z_sensor = self.gcn_sensor(sensor_features, adj_sensor) @ sensor_features.transpose(1, 2)
        z_aging = self.gcn_aging(aging_features, adj_aging) @ sensor_features.transpose(1, 2)

        # Cross-view consistency
        loss_cvgcl = consistency_loss(z_sensor, z_aging)

        # Fusion
        # enc_out = (z_sensor + z_aging) / 2  # (B, T, d_model)

        fusion_layer = FusionMessagePassing(self.input_feature).to(self.device)
        enc_out = fusion_layer(z_sensor, z_aging)  # (B, T, d_model)

        # Prediction
        fc_output = self.decoder(enc_out)
        fc_output = self.decoder(fc_output.permute(0, 2, 1))


        rul_prediction = torch.abs(fc_output.squeeze(-1))
        if return_loss:
            return loss_cvgcl * lambda_cvgcl, rul_prediction
        else:
            return None, rul_prediction



class MLPFusionMessagePassing(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, z_sensor, z_aging):
        combined = torch.cat([z_sensor, z_aging], dim=-1)  # concat views
        msg = self.mlp(combined)
        return z_sensor + z_aging + msg  # residual fusion


class FusionMessagePassing(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_sensor = nn.Linear(d_model, d_model)
        self.gate_aging  = nn.Linear(d_model, d_model)
        self.fc = nn.Bilinear(d_model, d_model, d_model)

    def forward(self, z_sensor, z_aging):
        # Compute gates
        gate_s = torch.sigmoid(self.gate_sensor(z_aging))  # how much aging influences sensor
        gate_a = torch.sigmoid(self.gate_aging(z_sensor))  # how much sensor influences aging

        # Message passing
        sensor_msg = z_sensor + gate_s * z_aging
        aging_msg  = z_aging + gate_a * z_sensor

        # Fusion
        return sensor_msg + aging_msg
