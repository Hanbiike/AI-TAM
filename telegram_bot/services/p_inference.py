import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================================================
# УТИЛИТЫ И АРХИТЕКТУРА
# =====================================================
def sim_x(q_cat, q_num, X_cat, X_num):
    cat_sim = (X_cat == q_cat.unsqueeze(0)).float()
    d = torch.abs(X_num - q_num.unsqueeze(0))
    scale = torch.abs(q_num.unsqueeze(0)) + 1e-6
    sim1 = 1.0 / (1.0 + d)
    sim2 = torch.exp(-d / scale)
    sim3 = torch.clamp(1.0 - d / scale, min=0.0)
    return torch.cat([cat_sim, sim1, sim2, sim3], dim=1)


def sim_xy(q_cat, q_num, q_y, X_cat, X_num, Y):
    Mx = sim_x(q_cat, q_num, X_cat, X_num)
    dy = torch.abs(Y - q_y)
    scale = torch.abs(q_y) + 1e-6
    y_sim1 = 1.0 / (1.0 + dy / scale)
    y_sim2 = torch.exp(-dy / scale)
    return torch.cat([Mx, y_sim1.unsqueeze(1), y_sim2.unsqueeze(1)], dim=1)


class ThreeLayerMoE(nn.Module):
    def __init__(self, d1, d2, n_samples, hidden=128, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.w1 = nn.Parameter(torch.ones(d1) * 0.1)
        self.w2 = nn.Parameter(torch.randn(n_heads, d2) * 0.1)
        self.sample_bias = nn.Parameter(torch.zeros(n_samples))
        self.gating = nn.Sequential(nn.Linear(d2, hidden), nn.ReLU(), nn.Linear(hidden, n_heads))
        self.residual_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_heads)
        ])


# =====================================================
# КЛАСС ИНФЕРЕНСА
# =====================================================
class PricePredictor:
    def __init__(self, model_path: str, context_path: str, device="cpu"):
        self.device = device
        # Вычисляем путь к корню (C:\проекты\aitam)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        full_model_path = os.path.join(base_dir, model_path)
        full_context_path = os.path.join(base_dir, context_path)

        print(f"📥 Загрузка контекста из: {full_context_path}")
        ctx = joblib.load(full_context_path)

        self.X = ctx["X"]
        self.cat_cols = ctx["cat_cols"]
        self.num_cols = ctx["num_cols"]
        self.y_tr_np = np.asarray(ctx["y"])
        self.default_values = self.X.median(numeric_only=True).to_dict()

        self.X_cat_tr = torch.tensor(self.X[self.cat_cols].values, dtype=torch.float32, device=self.device)
        self.X_num_tr = torch.tensor(self.X[self.num_cols].values, dtype=torch.float32, device=self.device)
        self.y_tr = torch.tensor(self.y_tr_np, dtype=torch.float32, device=self.device)

        n_samples = len(self.X)
        d1 = sim_x(self.X_cat_tr[0], self.X_num_tr[0], self.X_cat_tr, self.X_num_tr).shape[1]
        d2 = sim_xy(self.X_cat_tr[0], self.X_num_tr[0], torch.tensor(0.0, device=self.device),
                    self.X_cat_tr, self.X_num_tr, self.y_tr).shape[1]

        self.model = ThreeLayerMoE(d1=d1, d2=d2, n_samples=n_samples).to(self.device)

        # Загрузка весов с очисткой от _orig_mod
        state_dict = torch.load(full_model_path, map_location=self.device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        self.model.eval()
        print("🔥 Модель успешно готова к инференсу!")

    @torch.no_grad()
    def predict(self, user_data: dict, g: float = 0.5) -> float:
        input_row = self.default_values.copy()
        for k, v in user_data.items():
            if k in input_row:
                input_row[k] = v

        df_user = pd.DataFrame([input_row])
        X_cat_te = torch.tensor(df_user[self.cat_cols].values, dtype=torch.float32, device=self.device)[0]
        X_num_te = torch.tensor(df_user[self.num_cols].values, dtype=torch.float32, device=self.device)[0]

        M1 = sim_x(X_cat_te, X_num_te, self.X_cat_tr, self.X_num_tr)
        s1 = M1 @ self.model.w1
        a1 = torch.softmax(s1, dim=0)
        y1 = (a1 * self.y_tr).sum()

        M2 = sim_xy(X_cat_te, X_num_te, y1, self.X_cat_tr, self.X_num_tr, self.y_tr)
        scores2 = M2 @ self.model.w2.T + self.model.sample_bias[:, None]
        a2 = torch.softmax(scores2, dim=0)

        y_heads = (a2 * self.y_tr.unsqueeze(1)).sum(dim=0)
        feats_heads = (a2.unsqueeze(2) * M2.unsqueeze(1)).sum(dim=0)

        feats_global = feats_heads.mean(dim=0)
        g_weights = torch.softmax(self.model.gating(feats_global), dim=0)
        residual_heads = torch.stack(
            [self.model.residual_mlps[h](feats_heads[h]) for h in range(self.model.n_heads)]).squeeze()
        soft_weights = torch.softmax(-torch.abs(residual_heads), dim=0)

        final_weights = g * g_weights + (1 - g) * soft_weights
        final_weights = final_weights / (final_weights.sum() + 1e-8)

        y2 = (final_weights * y_heads).sum()
        residual = (final_weights * residual_heads).sum()

        area = float(user_data.get('area', input_row.get('area', 50)))
        return round((y2 + residual).item() * area, 2)