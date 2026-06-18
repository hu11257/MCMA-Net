import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMotifScanner(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1), nn.BatchNorm1d(out_channels), nn.GELU())
        self.conv7 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 7, padding=3), nn.BatchNorm1d(out_channels), nn.GELU())
        self.conv11 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 11, padding=5), nn.BatchNorm1d(out_channels), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv1d(out_channels * 3, out_channels * 2, 1), nn.BatchNorm1d(out_channels * 2), nn.GELU())

    def forward(self, x):
        return self.fuse(torch.cat([self.conv3(x), self.conv7(x), self.conv11(x)], dim=1))


class PhysicalDescriptor(nn.Module):
    def __init__(self, in_channels=6, out_channels=128):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, out_channels, 3, padding=1), nn.BatchNorm1d(out_channels), nn.GELU())

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DualCrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.attn_seq2phy = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_seq2phy = nn.LayerNorm(embed_dim)
        self.attn_phy2seq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_phy2seq = nn.LayerNorm(embed_dim)

    def forward(self, h_seq, h_phy):
        fused1, _ = self.attn_seq2phy(h_seq, h_phy, h_phy, need_weights=False)
        fused2, _ = self.attn_phy2seq(h_phy, h_seq, h_seq, need_weights=False)
        return self.norm_seq2phy(h_seq + fused1), self.norm_phy2seq(h_phy + fused2)


class GatedFusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim * 4, embed_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, h_seq, h_phy, h_fused1, h_fused2):
        gate = self.gate(torch.cat([h_seq, h_phy, h_fused1, h_fused2], dim=-1))
        return self.out_norm(gate * h_fused1 + (1.0 - gate) * h_fused2)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)


class MCMA_Net(nn.Module):
    def __init__(self, seq_channels=4, phy_channels=6, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.seq_scanner = MultiScaleMotifScanner(in_channels=seq_channels, out_channels=64)
        self.phy_descriptor = PhysicalDescriptor(in_channels=phy_channels, out_channels=hidden_dim)
        self.dual_cross_attn = DualCrossAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout)
        self.gated_fusion = GatedFusion(embed_dim=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_pooling = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, seq_onehot, shape, cons):
        phy_input = torch.cat([shape[:, :5, :], cons], dim=1)
        h_seq = self.seq_scanner(seq_onehot).permute(0, 2, 1).contiguous()
        h_phy = self.phy_descriptor(phy_input).permute(0, 2, 1).contiguous()
        h_fused1, h_fused2 = self.dual_cross_attn(h_seq, h_phy)
        h_final = self.gated_fusion(h_seq, h_phy, h_fused1, h_fused2)
        context_feat = self.transformer_encoder(h_final)
        global_feat = self.attn_pooling(context_feat)
        return self.classifier(global_feat)
