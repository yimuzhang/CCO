import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class CausalDataset(Dataset):
    def __init__(self, n_samples, d, w_true, e_y, c_vec, e_z_vec, e_x_vec, mu_x_vec=None, mu_z_vec=None, device='cpu'):
        self.n_samples = n_samples
        self.d = d
        self.w_true = w_true.to(device)
        self.e_y = e_y
        self.c_vec = c_vec.to(device)
        self.e_z_vec = e_z_vec.to(device)
        self.e_x_vec = e_x_vec.to(device)
        self.mu_x_vec = torch.zeros(d, device=device) if mu_x_vec is None else torch.as_tensor(mu_x_vec, dtype=torch.float32, device=device)
        self.mu_z_vec = torch.zeros(d, device=device) if mu_z_vec is None else torch.as_tensor(mu_z_vec, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = torch.randn(self.d, device=self.device) * torch.sqrt(self.e_x_vec) + self.mu_x_vec
        eps_y = torch.randn(1, device=self.device) * torch.sqrt(torch.tensor(self.e_y, device=self.device))
        Y = (X * (self.w_true ** 2)).sum() + eps_y
        eps_z = torch.randn(self.d, device=self.device) * torch.sqrt(self.e_z_vec) + self.mu_z_vec
        Z = self.c_vec * Y + eps_z
        return X, Z, Y

def train_model(
    w_true=None,
    env_params_list=None,
    T=10000,
    eta_0=1e-2,
    batch_size=128,
    d=4,
    w_initial=0.01,
    gamma=1.0,
    seed=42,
    device=None,
):
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if w_true is None:
        w_true = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32, device=device)
    else:
        w_true = w_true.to(device)

    if env_params_list is None:
        env_params_list = [
            {
                "n_samples": 100000,
                "e_y": 1.0,
                "c_vec": torch.tensor([0.0, 1.2, 1.2]),
                "e_z_vec": torch.ones(d),
                "e_x_vec": torch.ones(d),
                "mu_x_vec": torch.zeros(d),
                "mu_z_vec": torch.zeros(d),
            }
        ]

    loaders, iterators = [], []
    for env in env_params_list:
        ds = CausalDataset(
            n_samples=env["n_samples"],
            d=d,
            w_true=w_true,
            e_y=env["e_y"],
            c_vec=env["c_vec"],
            e_z_vec=env["e_z_vec"],
            e_x_vec=env["e_x_vec"],
            mu_x_vec=env.get("mu_x_vec", None),
            mu_z_vec=env.get("mu_z_vec", None),
            device=device,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        loaders.append(loader)
        iterators.append(iter(loader))

    w = torch.ones(d, device=device) * w_initial
   # w = w_true*0.5#+torch.randn(d,device=device) * w_initial*10
    w[0] = 0.0
    v = torch.zeros(d, device=device)
    key = torch.zeros(d, device=device)
    value = torch.zeros(d, device=device)
    w_hist, v_hist = [], []
    pm_hist,ps_hist = [], []

    decay_every = max(1, int(T / np.log2(T)))
    p_final = None
    for t in range(T):
        decay_factor = t // decay_every
        eta_t = eta_0 * (0.5 ** decay_factor)
        env_idx = np.random.randint(len(loaders))
        try:
            X, Z, Y = next(iterators[env_idx])
        except StopIteration:
            iterators[env_idx] = iter(loaders[env_idx])
            X, Z, Y = next(iterators[env_idx])

        X, Z, Y = X.to(device), Z.to(device), Y.to(device)
        quad = w ** 2
        vx = v @ X.T
        vz = v @ Z.T
        logits_q = torch.stack([vx, vz], dim=1) 
        kx = key @ X.T
        kz = key @ Z.T
        logits_k = torch.stack([kx, kz], dim=1) 
        val_x = value @ X.T
        val_z = value @ Z.T
        logits_v = torch.stack([val_x, val_z], dim=1) 
        logits=logits_q@logits_k.T
        p = torch.softmax(logits, dim=1)[:, 0]
        mix = p.unsqueeze(1) * X + (1 - p.unsqueeze(1)) * Z
        pred = (mix * quad).sum(dim=1)
        error = pred - Y.squeeze()

        grad_w = (error.unsqueeze(1) * mix * w).mean(dim=0)
        w = w - eta_t * grad_w

        diff = X - Z
        factor = error * (quad @ diff.T) * p * (1 - p)
        grad_v = (factor.unsqueeze(1) * diff).mean(dim=0)
        v = v - eta_t * gamma * grad_v

        #w_hist.append(torch.norm(w-w_true).detach().cpu().numpy())
        w_hist.append(w.detach().cpu().numpy())
        v_hist.append(v.detach().cpu().numpy())

        # 记录最后一个batch的p的均值
        p_mean = p.mean().item()
        p_std = p.std().item()
        pm_hist.append(p_mean)
        ps_hist.append(p_std)
        if t == T - 1:
            p_final = p_mean
        if (t + 1) % 500 == 0 or t == 0:
            print(f"[{t+1}/{T}] η={eta_t:.2e}  w={w.cpu().numpy()}  v={v.cpu().numpy()}  p_mean={p_mean:.6f}  p_std={p_std:.6f}")

    return w.detach().clone(), v.detach().clone(), w_hist, v_hist, p_final,pm_hist,ps_hist

def evaluate_on_env(env, w_final, v_final, w_true, test_m=10000, gamma=100.0, device="cpu", p_override=None):
    d = w_final.shape[0]
    w_final = w_final.to(device)
    v_final = v_final.to(device)
    w_true = w_true.to(device)
    e_x_vec = env["e_x_vec"].to(device)
    mu_x_vec = env.get("mu_x_vec", torch.zeros(d)).to(device)
    X_test = torch.randn(test_m, d, device=device) * torch.sqrt(e_x_vec) + mu_x_vec

    eps_y_test = torch.randn(test_m, device=device) * torch.sqrt(torch.tensor(env["e_y"], device=device))
    Y_test = (X_test * (w_true**2)).sum(dim=1) + eps_y_test

    e_z_vec = env["e_z_vec"].to(device)
    mu_z_vec = env.get("mu_z_vec", torch.zeros(d)).to(device)
    eps_z_test = torch.randn(test_m, d, device=device) * torch.sqrt(e_z_vec) + mu_z_vec
    c_vec = env["c_vec"].to(device)
    Z_test = c_vec * Y_test.unsqueeze(1) + eps_z_test

    quad_final = w_final ** 2

    if p_override is None:
        vx_test = v_final  @ X_test.T
        vz_test = v_final  @ Z_test.T
        logits_test = torch.stack([vx_test, vz_test], dim=1) * gamma
        p_test = torch.softmax(logits_test, dim=1)[:, 0].unsqueeze(1)
    else:
        p_test = torch.full((test_m, 1), p_override, device=device)

    pred_test = (p_test * (X_test * quad_final)).sum(dim=1) + ((1 - p_test) * (Z_test * quad_final)).sum(dim=1)
    mse = ((pred_test - Y_test) ** 2).mean().item()
    p_mean = p_test.mean().item()
    p_std = p_test.std().item()
    return mse, p_mean, p_std

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dim=10
    c_vec=torch.ones(n_dim)*0.1
    c_vec[0]=0
    e_z_vec=abs(torch.randn(n_dim))
    e_z_vec[0]=0
    e_x_vec=abs(torch.randn(n_dim))
    e_x_vec[0]=0
    e_x_vec[1]=e_x_vec[1]*100
    mu_x_vec=torch.zeros(n_dim)
    mu_x_vec[0]=100
    mu_z_vec=torch.zeros(n_dim)
    envs = [
        {
            "n_samples": 100000,
            "e_y": 1.0,
            "c_vec": c_vec,
            "e_z_vec": e_z_vec,
            "e_x_vec": e_x_vec,
            "mu_x_vec": mu_x_vec,
            "mu_z_vec": mu_z_vec,
        },
        {
            "n_samples": 100000,
            "e_y": 1.0,
            "c_vec": c_vec,
            "e_z_vec": e_z_vec,
            "e_x_vec": e_x_vec,
            "mu_x_vec": mu_x_vec,
            "mu_z_vec": mu_z_vec,
        }
    ]
    # envs = [
    #     {
    #         "n_samples": 100000,
    #         "e_y": 1.0,
    #         "c_vec": torch.tensor([0, 0.1, 0.1, 0.1]),
    #         "e_z_vec": torch.tensor([0, 1, 1, 1]),
    #         "e_x_vec": torch.tensor([0, 10, 0.8, 0.8]),
    #         "mu_x_vec": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    #         "mu_z_vec": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    #     },
    #     {
    #         "n_samples": 100000,
    #         "e_y": 1.0,
    #         "c_vec": torch.tensor([0, 0.1, 0.1, 0.1]),
    #         "e_z_vec": torch.tensor([0, 1, 1, 1]),
    #         "e_x_vec": torch.tensor([0, 10, 0.8, 0.8]),
    #         "mu_x_vec": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    #         "mu_z_vec": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    #     }
    # ]

    c_vec=torch.randn(n_dim)
    c_vec[0]=0
    e_z_vec=abs(torch.randn(n_dim))
    e_z_vec[0]=0
    e_x_vec=abs(torch.randn(n_dim))
    e_x_vec[0]=0
    e_x_vec[1]=e_x_vec[1]*50
    mu_x_vec=torch.zeros(n_dim)
    mu_x_vec[0]=5
    mu_z_vec=torch.zeros(n_dim)
    mu_z_vec[0]=0
    test_envs = [
        {
            "n_samples": 100000,
            "e_y": 1.0,
           "c_vec": c_vec,
            "e_z_vec": e_z_vec,
            "e_x_vec": e_x_vec,
            "mu_x_vec": mu_x_vec,
            "mu_z_vec": mu_z_vec,
        },
        {
            "n_samples": 100000,
            "e_y": 0.5,
           "c_vec": c_vec,
            "e_z_vec": e_z_vec,
            "e_x_vec": e_x_vec,
            "mu_x_vec": mu_x_vec,
            "mu_z_vec": mu_z_vec,
        }
    ]
    #w_true = torch.tensor([0.0, 2.0, 1.5, 1.0,0.5])
    w_true = torch.ones(n_dim)*0.6#abs(torch.randn(n_dim))*2
    w_true[0]=0
    print('wtrue:',w_true)
    seed = 3407

    w_final, v_final, w_hist, v_hist, p_final, pm, ps = train_model(
        w_true=w_true,
        env_params_list=envs,
        T=10000,
        eta_0=4e-2,
        batch_size=64,
        d=n_dim,
        w_initial=0.1,
        gamma=20.0,
        seed=seed,
        device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./causal_run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "w.npy"), w_hist)
    np.save(os.path.join(save_dir, "v.npy"), v_hist)
    pd.DataFrame(w_hist).to_csv(os.path.join(save_dir, "w.csv"), index=False)
    pd.DataFrame(v_hist).to_csv(os.path.join(save_dir, "v.csv"), index=False)
    pd.DataFrame(pm).to_csv(os.path.join(save_dir, "pm.csv"), index=False)
    pd.DataFrame(ps).to_csv(os.path.join(save_dir, "ps.csv"), index=False)

    print("\n========== Test Results ==========")
    print(f"Final train p_mean (p_final): {p_final:.6f}")
    mse, p_mean, p_std = evaluate_on_env(envs[0], w_final, v_final, w_true, test_m=10000, gamma=100.0, device=device, p_override=p_final)
    print(f"Train Env0:  MSE={mse:.6f}, p_mean={p_mean:.6f}, p_std={p_std:.6f}")
    for i, test_env in enumerate(test_envs):
        mse, p_mean, p_std = evaluate_on_env(test_env, w_final, v_final, w_true, test_m=10000, gamma=100.0, device=device, p_override=p_final)
        print(f"Test Env{i+1}: MSE={mse:.6f}, p_mean={p_mean:.6f}, p_std={p_std:.6f}")

    with open(os.path.join(save_dir, "test_result.txt"), "w") as f:
        f.write(f"Random seed   : {seed}\n")
        f.write(f"p_final       : {p_final:.6f}\n")
        mse, p_mean, p_std = evaluate_on_env(envs[0], w_final, v_final, w_true, test_m=10000, gamma=100.0, device=device, p_override=p_final)
        f.write(f"Train Env0:  MSE={mse:.6f}, p_mean={p_mean:.6f}, p_std={p_std:.6f}\n")
        for i, test_env in enumerate(test_envs):
            mse, p_mean, p_std = evaluate_on_env(test_env, w_final, v_final, w_true, test_m=10000, gamma=100.0, device=device, p_override=p_final)
            f.write(f"Test Env{i+1}: MSE={mse:.6f}, p_mean={p_mean:.6f}, p_std={p_std:.6f}\n")
    print("结果保存在目录：", save_dir)
