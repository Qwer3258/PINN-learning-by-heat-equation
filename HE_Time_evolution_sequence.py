# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:12:29 2026

@author: LenovoTest
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L = 1.0
T_end = 1.0

TRAIN_ALPHAS = [0.05, 0.1, 0.2]

TEST_ALPHA = 0.15  

EPOCHS = 15000
LR = 1e-3
N_PDE = 3000
N_BC = 200
N_IC = 200
N_DATA = 300

def exact_solution(x, t, alpha):
    return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

class ParametricPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ParametricPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) 
        )
    
    def forward(self, x, t, alpha):
        inputs = torch.cat([x, t, alpha], dim=1)
        return self.net(inputs)

def generate_training_data():
    data_dict = {}
    
    data_x, data_t, data_a, data_u = [], [], [], []
    
    for a in TRAIN_ALPHAS:
        x_grid = np.linspace(0, L, 50)
        t_grid = np.linspace(0, T_end, 30) 
        
        # Expend the x, t list into 2 dimension matrix 
        X, T = np.meshgrid(x_grid, t_grid) 
        
        # Convert X and T into solution function 
        U = exact_solution(X, T, a)
        
        # 30*50 = 1500 
        total_points = X.size 
        
        # Pick the training data (300/1500) 
        idx = np.random.choice(total_points, N_DATA, replace=False)
        
        # flat: Flat the matrix into one dimension 
        for i in idx:
            data_x.append(X.flat[i])
            data_t.append(T.flat[i])
            data_a.append(a)
            data_u.append(U.flat[i])
    
    # -1: Inferring the size; 1: Converting the vector into 2D [N, 1]
    data_dict['x'] = torch.tensor(data_x, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['t'] = torch.tensor(data_t, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['a'] = torch.tensor(data_a, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['u'] = torch.tensor(data_u, dtype=torch.float32).view(-1, 1).to(device)
    
    bc_x, bc_t, bc_a, bc_u = [], [], [], []
    for a in TRAIN_ALPHAS: 
        bc_x.extend([0.0]* (N_BC//2) + [L]* (N_BC//2))
        bc_t.extend(np.random.rand(N_BC).tolist())
        bc_a.extend([a]* N_BC)
        bc_u.extend([0.0]* N_BC)
    
    data_dict['bc_x'] = torch.tensor(bc_x, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['bc_t'] = torch.tensor(bc_t, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['bc_a'] = torch.tensor(bc_a, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['bc_u'] = torch.tensor(bc_u, dtype=torch.float32).view(-1, 1).to(device)
    
    ic_x, ic_t, ic_a, ic_u = [], [], [], []
    for a in TRAIN_ALPHAS:
        ic_x.extend(np.random.rand(N_IC).tolist())
        ic_t.extend([0.0]* N_IC)
        ic_a.extend([a]* N_IC)
        ic_u.extend(np.sin(np.pi * np.array(ic_x[-N_IC:])).tolist())
    
    data_dict['ic_x'] = torch.tensor(ic_x, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['ic_t'] = torch.tensor(ic_t, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['ic_a'] = torch.tensor(ic_a, dtype=torch.float32).view(-1, 1).to(device)
    data_dict['ic_u'] = torch.tensor(ic_u, dtype=torch.float32).view(-1, 1).to(device)
    
    return data_dict

def compute_loss(model, data_dict, alpha_pde):
    x_f = torch.rand(N_PDE, 1, device=device, requires_grad=True)
    t_f = torch.rand(N_PDE, 1, device=device, requires_grad=True)
    a_f = torch.ones(N_PDE, 1, device=device) * alpha_pde
    
    u_f = model(x_f, t_f, a_f)
    
    u_t = torch.autograd.grad(
        u_f, t_f,
        grad_outputs=torch.ones_like(u_f),
        create_graph=True
    )[0]
    
    u_x = torch.autograd.grad(
        u_f, x_f,
        grad_outputs=torch.ones_like(u_f),
        create_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        u_x, x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]
    
    residual = u_t - a_f * u_xx
    loss_pde = torch.mean(residual**2)
    
    u_bc_pred = model(data_dict['bc_x'], data_dict['bc_t'], data_dict['bc_a'])
    loss_bc = torch.mean((u_bc_pred - data_dict['bc_u'])**2)
    
    u_ic_pred = model(data_dict['ic_x'], data_dict['ic_t'], data_dict['ic_a'])
    loss_ic = torch.mean((u_ic_pred - data_dict['ic_u'])**2)
    
    u_data_pred = model(data_dict['x'], data_dict['t'], data_dict['a'])
    loss_data = torch.mean((u_data_pred - data_dict['u'])**2)
    
    total_loss = loss_pde + loss_bc + loss_ic + loss_data
    
    return total_loss, loss_pde, loss_bc, loss_ic, loss_data

def train():
    model = ParametricPINN(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    data_dict = generate_training_data()
    
    print("Start training...")
    print(f"Training alpha set: {TRAIN_ALPHAS}")
    print(f"Testing alpha: {TEST_ALPHA}")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        alpha_pde = np.random.choice(TRAIN_ALPHAS)
        
        total_loss, loss_pde, loss_bc, loss_ic, loss_data = compute_loss(
            model, data_dict, alpha_pde
        )
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | Total: {total_loss.item():.3e} | "
                  f"PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | "
                  f"IC: {loss_ic.item():.3e} | Data: {loss_data.item():.3e} | "
                  f"PDE_alpha: {alpha_pde:.2f}")
    
    return model

# Time evolution sequuence image
def plot_time_evolution(model, alpha_test, title_suffix=""):
    model.eval()
    
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_test = np.linspace(0, L, 100)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    max_error = 0
    
    for idx, ti in enumerate(times):
        x_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)
        t_tensor = torch.ones(100, 1, device=device) * ti
        a_tensor = torch.ones(100, 1, device=device) * alpha_test
        
        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor, a_tensor).cpu().numpy().flatten()
        
        u_true = exact_solution(x_test, ti, alpha_test)
        
        error = np.abs(u_pred - u_true)
        max_error = max(max_error, error.max())
        l2_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        
        ax = axes[idx]
        ax.plot(x_test, u_true, 'b-', linewidth=2, label='Exact (Analytical)')
        ax.plot(x_test, u_pred, 'r--', linewidth=2, label='PINN Prediction')
        ax.fill_between(x_test, u_true, u_pred, alpha=0.2, color='gray', label=f'Error')
        ax.set_title(f't = {ti:.1f}, L2 Error = {l2_err*100:.2f}%', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'Time Evolution Sequence (α = {alpha_test}) {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n[α = {alpha_test}] Max absolute error: {max_error:.6e}")
    return max_error

if __name__ == '__main__':
    model = train()
    
    print("\n" + "="*50)
    print("Verification 1: Training alpha = 0.1")
    plot_time_evolution(model, alpha_test=0.1, title_suffix="[In-distribution]")
    
    print("\n" + "="*50)
    print(f"Verification 2：Test alpha (Untrained) = {TEST_ALPHA}")
    plot_time_evolution(model, alpha_test=TEST_ALPHA, title_suffix="[Out-of-distribution]")
    
    print("\n" + "="*50)
    print("Verification 3：Training alpha = 0.05")
    plot_time_evolution(model, alpha_test=0.05, title_suffix="[Boundary of training set]") 
    
    alpha_save = 0.1   
    nx, nt = 100, 100
    
    x_grid = np.linspace(0, L, nx)
    t_grid = np.linspace(0, T_end, nt)
    X_save, T_save = np.meshgrid(x_grid, t_grid)
    
    x_tensor = torch.tensor(X_save.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(T_save.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    a_tensor = torch.ones(nx * nt, 1, device=device) * alpha_save
    
    model.eval()
    with torch.no_grad():
        U_save = model(x_tensor, t_tensor, a_tensor).cpu().numpy()
    
    noise = 0.01 * np.std(U_save) * np.random.randn(*U_save.shape)
    U_save = U_save + noise
    
    np.savez('forward_data.npz',
             x=X_save.flatten(),
             t=T_save.flatten(),
             u=U_save.flatten(),
             true_alpha=alpha_save)
    
    print(f"Saved {len(X_save.flatten())} points for alpha = {alpha_save} to forward_data.npz") 