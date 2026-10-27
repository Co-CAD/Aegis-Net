# -*- coding: utf-8 -*-
"""增强版 1D CNN-LSTM（早期检测 N<=15 优先）。

相对原始 gbs_crispr_cnn_lstm_model_implementation.py（保持不动）的改动：
  P0 特征融合：把 22 维手工动力学特征（RF 的杀手锏）经独立 MLP 分支注入，与 LSTM 读出拼接。
  P1 读出升级：双向 LSTM + 时间维 mean&max 池化，替掉"只取最后一个时刻"。
  P3 双通道输入：通道0=原始曲线，通道1=一阶差分 Δ（显式暴露斜率）。
     卷积加到两层、加宽到 64（感受野 3->5）。

训练侧增强（label smoothing / cosine LR / K-seed 集成）放在评测脚本 cnn_lstm_enhanced_cv.py。
本文件只定义模型与特征提取，便于复用与单测。
"""
import numpy as np
import torch
import torch.nn as nn

N_FEATS = 22


# ---------------------------------------------------------------------
# 手工动力学特征（对 N>=1 全部有定义；不足时退化为 0）。在原始幅度上计算，
# 标准化在评测脚本里只用训练子集拟合（无泄露）。
# ---------------------------------------------------------------------
def _feat_one(s):
    s = np.asarray(s, dtype=np.float64)
    N = len(s)
    eps = 1e-8
    mean, std, mn, mx = s.mean(), s.std(), s.min(), s.max()
    rng, final, first = mx - mn, s[-1], s[0]
    if N >= 2:
        t = np.arange(N, dtype=np.float64)
        slope = float(np.polyfit(t, s, 1)[0])
        d = np.diff(s)
        d_mean, d_std, d_max, d_min = d.mean(), d.std(), d.max(), d.min()
        rise = float(d[d > 0].sum())
        fall = float(-d[d < 0].sum())
        frac_rise = float((d > 0).mean())
        half = N // 2
        late_early = float(s[half:].mean() - s[:max(half, 1)].mean())
    else:
        slope = d_mean = d_std = d_max = d_min = rise = fall = frac_rise = late_early = 0.0
    curv = float(np.diff(s, 2).mean()) if N >= 3 else 0.0
    sat = float(final / (mx + eps))
    fz = float((final - mean) / (std + eps))
    amax = float(np.argmax(s) / max(N - 1, 1))
    area = float((s - first).mean())
    return [mean, std, mn, mx, rng, final, first, final - first, slope,
            d_mean, d_std, d_max, d_min, rise, fall, frac_rise,
            curv, sat, fz, amax, area, late_early]


def extract_features(X, N):
    """X:[n,120] -> [n,22] float32，用前 N 个循环。"""
    F = [_feat_one(X[i, :N]) for i in range(len(X))]
    return np.asarray(F, dtype=np.float32)


def make_channels(seq):
    """seq:[n,N] -> (raw[n,N], diff[n,N])，diff 为一阶差分（首位补 0，保持长度 N）。"""
    raw = seq.astype(np.float32)
    diff = np.zeros_like(raw)
    if seq.shape[1] >= 2:
        diff[:, 1:] = np.diff(seq, axis=1)
    return raw, diff


# ---------------------------------------------------------------------
# 增强模型
# ---------------------------------------------------------------------
class CRISPR_CNNLSTM_Enhanced(nn.Module):
    def __init__(self, in_channels=2, n_feats=N_FEATS, cnn_filters=64,
                 lstm_hidden=64, feat_hidden=64, num_classes=2, p_drop=0.3):
        super().__init__()
        # 双层一维卷积（局部斜率/曲率特征，感受野 5）
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        # 双向 LSTM
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        lstm_out = lstm_hidden * 2          # 双向
        readout = lstm_out * 2              # mean + max 拼接

        # 手工特征独立 MLP 分支
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feats, feat_hidden), nn.BatchNorm1d(feat_hidden),
            nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(feat_hidden, feat_hidden), nn.ReLU(inplace=True),
        )
        # 融合分类头
        self.head = nn.Sequential(
            nn.Linear(readout + feat_hidden, 64), nn.ReLU(inplace=True),
            nn.Dropout(p_drop), nn.Linear(64, num_classes),
        )

    def forward(self, x_seq, x_feat):
        # x_seq:[B,C,N]  x_feat:[B,F]
        h = self.conv(x_seq)               # [B, cnn_filters, N]
        h = h.transpose(1, 2)              # [B, N, cnn_filters]
        out, _ = self.lstm(h)              # [B, N, 2*lstm_hidden]
        mean = out.mean(dim=1)
        mx = out.max(dim=1).values
        readout = torch.cat([mean, mx], dim=1)
        f = self.feat_mlp(x_feat)
        z = torch.cat([readout, f], dim=1)
        return self.head(z)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CRISPR_CNNLSTM_Enhanced()
    print("params:", count_params(m))
    xs = torch.randn(4, 2, 10)
    xf = torch.randn(4, N_FEATS)
    print("out:", m(xs, xf).shape)
