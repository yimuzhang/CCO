import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ----------------------------
# 1. 数据生成（同前）
# ----------------------------
def generate_data(n_samples: int, d_x=8, d_z=8, spurious=True, seed=0,wz=0.1):
    np.random.seed(seed)
    x = np.random.randn(n_samples, d_x)
    y = np.average(x+np.random.randn(*x.shape)*0.001,axis=1)
    if spurious:
        z = wz* np.ones((n_samples, d_z)) * (1 - y)[:, np.newaxis]
    else:
        z = wz*np.ones((n_samples, d_z)) * (1 - y)[:, np.newaxis]
    z += np.random.randn(*z.shape) * 3
    X = np.concatenate([x, z], axis=1).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

# ----------------------------
# 2. 自定义 Transformer Layer（返回 attention weights）
# ----------------------------
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        # Self-attention
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True  # ← 关键：返回 attention weights
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights  # (B, L, L)


class StandardTransformerWithAttn(nn.Module):
    def __init__(self, d_x=8, d_z=8, d_model=8, nhead=2, num_layers=2):
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_model = d_model

        self.proj_x = nn.Linear(d_x, d_model)
        self.proj_z = nn.Linear(d_z, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model))

        # Use custom layers to get attention
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X, return_attention=False):
        x = X[:, :self.d_x]
        z = X[:, self.d_x:]
        x_emb = self.proj_x(x)
        z_emb = self.proj_z(z)
        seq = torch.stack([x_emb, z_emb], dim=1)  # (B, 2, d_model)
        seq = seq + self.pos_emb.unsqueeze(0)

        attn_weights_list = []
        out = seq
        for layer in self.layers:
            out, attn_w = layer(out)
            if return_attention:
                attn_weights_list.append(attn_w)  # each: (B, H, L, L)

        pooled = out.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)

        if return_attention:
            # Return average attention over heads and layers for the last token's query or overall mean
            # We'll compute mean attention from all queries to all keys
            # Shape after mean: (B, 2)
            avg_attn = torch.stack(attn_weights_list).mean(dim=[0, 1])  # (B, 2, 2) → mean over layers & heads
            # For simplicity, use average attention **to each key** (i.e., column-wise mean per sample)
            # Or: take attention from a specific query (e.g., first token)
            # Here we return full (B, 2, 2), but for plotting we can average over B
            return logits, avg_attn
        else:
            return logits

# ----------------------------
# 3. 训练函数（无 attention）
# ----------------------------
def visualize_attention(model, X_sample, token_names=["x (causal)", "z (spurious)"]):
    model.eval()
    with torch.no_grad():
        _, attn = model(X_sample, return_attention=True)  # attn: (B, 2, 2)

    # Average over batch
    avg_attn = attn.mean(dim=0).cpu().numpy()  # (2, 2)
   # print(avg_attn)
    return avg_attn

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # # Plot: attention FROM each query TO keys
    # for i, name in enumerate(token_names):
    #     axes[0].bar(token_names, avg_attn[i], alpha=0.7, label=f"Query: {name}")
    # axes[0].set_title("Attention FROM each token (query)")
    # axes[0].set_ylabel("Attention Weight")
    # axes[0].legend()

    # # Plot: average attention TO each key (column mean)
    # to_x = avg_attn[0].mean()
    # to_z = avg_attn[ 1].mean()
    # axes[1].bar(["x (causal)", "z (spurious)"], [to_x, to_z], color=['steelblue', 'orange'])
    # axes[1].set_title("Average Attention TO each token (key)")
    # axes[1].set_ylabel("Mean Attention Weight")

    # plt.tight_layout()
    # plt.show()

    # print("\n📊 Average Attention Matrix (rows=query, cols=key):")
    # print(f"          TO:   {'x':>10}   {'z':>10}")
    # for i, q in enumerate(token_names):
    #     print(f"FROM {q:10}: {avg_attn[i, 0]:10.3f}   {avg_attn[i, 1]:10.3f}")

def train_model(model, X_train, y_train, epochs=50, lr=2e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    attn_list=[]
    attn_list1=[]
    cnt=np.arange(epochs)
    loss_list=[]
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train.float())
        loss.backward()
        optimizer.step()
        attn=visualize_attention(model, X_train[:500])
        attn_list.append(attn[0])
        attn_list1.append(attn[1])
        loss_list.append(loss.detach())
        if (epoch + 1) % 100 == 0:
            acc = 0#accuracy_score(y_train.numpy(), (logits > 0).int().numpy())
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.3f}")
    #fig, ax1 = plt.subplots()

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.size'] = 20 
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.plot(cnt,attn_list,label='Attention on X')
    plt.plot(cnt,attn_list1,label='Attention on Z')
    x_line=100
    plt.axvline(x=x_line, color='green', linestyle='--', alpha=0.7, linewidth=2)

# 左侧标注'occupation'
    plt.text(90, 0.3, 'occupation', 
            ha='right', va='top', fontsize=16, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 右侧标注'crowding out'
    plt.text(110, 0.3, 'crowding out', 
            ha='left', va='top', fontsize=16, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    # for label in ax1.get_yticklabels():
    #     label.set_color('black')

    # # 创建第二个共享X轴的坐标轴（右侧）
    # ax2 = ax1.twinx()

    # # 在第二个坐标轴上绘制数据，并设置颜色等属性
    # ax2.plot(cnt, loss_list, 'b-')  # 使用蓝色实线表示
    # ax2.set_ylabel('Loss', color='b')
    # for label in ax2.get_yticklabels():
    #     label.set_color('b')
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.size'] = 20 
    
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Attention', fontsize=20)
    plt.legend()
    plt.show()
# ----------------------------
# 4. 可视化 Attention
# ----------------------------

# ----------------------------
# 5. 主程序
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    d_x, d_z = 8, 8
    for xz in [0.01,0.1,0.5,1.0,2,5,10]:
        X_id, y_id = generate_data(2000, d_x=d_x, d_z=d_z, spurious=True, seed=0,wz=xz)
        

        model = StandardTransformerWithAttn(d_x=d_x, d_z=d_z, d_model=16, nhead=2, num_layers=2)

        print("Training model...")
        train_model(model, X_id, y_id, epochs=200, lr=5e-4)

        # Evaluate OOD
        model.eval()
        with torch.no_grad():
            for xzt in [0.01,0.1,0.5,1.0,2,5,10]:
                criterion = nn.BCEWithLogitsLoss()
                X_ood, y_ood = generate_data(1000, d_x=d_x, d_z=d_z, spurious=False, seed=1,wz=xzt)
                acc_ood = criterion(y_ood.numpy(), (model(X_ood) ).numpy())
                print(xzt,xz)
                print(f"\nOOD Accuracy: {acc_ood:.3f}")

    # Visualize attention on a batch of ID data
    print("\n🔍 Visualizing Attention on ID data...")
    visualize_attention(model, X_id[:500])  # use first 500 samples