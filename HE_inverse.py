# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:20:27 2026

@author: LenovoTest
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

data_file = 'forward_data.npz'  
data = np.load(data_file)

x_raw = data['x']
t_raw = data['t']
u_raw = data['u']

N_points = len(x_raw)

x_data = torch.tensor(x_raw, dtype=torch.float32).view(-1, 1)
t_data = torch.tensor(t_raw, dtype=torch.float32).view(-1, 1)
u_data = torch.tensor(u_raw, dtype=torch.float32).view(-1, 1)

N_sparse = 10000
idx = np.random.choice(len(x_data), N_sparse, replace=False)
x_data = x_data[idx]
t_data = t_data[idx]
u_data = u_data[idx] 

x_data = x_data.to(device)
t_data = t_data.to(device)
u_data = u_data.to(device)

print(f"Loaded {N_points} data points from {data_file}")
print(f"x range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
print(f"t range: [{t_raw.min():.3f}, {t_raw.max():.3f}]")

class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = InversePINN().to(device)

alpha_pred = nn.Parameter(torch.tensor([0.5], device = device))

optimizer = torch.optim.Adam(
    list(model.parameters()) + [alpha_pred],
    lr=1e-3
)

print(f"Initial alpha guess: {alpha_pred.item():.4f}")

alpha_history = []
loss_history = []

for epoch in range(30000):
    optimizer.zero_grad()
    
    x_f = torch.rand(1000, 1, device=device, requires_grad=True)
    t_f = torch.rand(1000, 1, device=device, requires_grad=True)
    
    u_f = model(x_f, t_f)
    u_t = torch.autograd.grad(u_f, t_f, torch.ones_like(u_f), create_graph=True)[0]
    u_x = torch.autograd.grad(u_f, x_f, torch.ones_like(u_f), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha_pred * u_xx
    loss_pde = torch.mean(residual**2)
    
    x_bc = torch.cat([torch.zeros(100, 1, device = device), torch.ones(100, 1, device = device)])
    t_bc = torch.rand(200, 1, device = device)
    loss_bc = torch.mean(model(x_bc, t_bc)**2)
    
    x_ic = torch.rand(100, 1, device = device)
    t_ic = torch.zeros(100, 1, device = device)
    u_ic_target = torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic_target)**2)
    
    u_pred_data = model(x_data, t_data)
    loss_data = torch.mean((u_pred_data - u_data)**2)
    
    lambda_pde = 1.0      
    lambda_bc  = 1.0      
    lambda_ic  = 1.0      
    lambda_data = 1.0  

    loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic + lambda_data * loss_data
    
    loss.backward()
    optimizer.step()
    
    alpha_history.append(alpha_pred.item())
    loss_history.append(loss.item())
    
    if epoch % 3000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | "
              f"alpha_pred: {alpha_pred.item():.6f} | "
              f"PDE: {loss_pde.item():.3e} | Data: {loss_data.item():.3e}")

print(f"\n{'='*60}")
print(f"INVERSE INFERENCE COMPLETE")
print(f"{'='*60}")
print(f"Inferred alpha : {alpha_pred.item():.6f}")

# If you saved true_alpha in the npz file, compare:
if 'true_alpha' in data:
    true_alpha = float(data['true_alpha'])
    print(f"True alpha     : {true_alpha:.6f}")
    print(f"Relative Error : {abs(alpha_pred.item() - true_alpha) / true_alpha * 100:.4f}%")

x_data_cpu = x_data.cpu()
t_data_cpu = t_data.cpu()
u_data_cpu = u_data.cpu()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(alpha_history, 'b-', linewidth=1.5)
if 'true_alpha' in data:
    plt.axhline(true_alpha, color='r', linestyle='--', label=f'True alpha')
plt.xlabel('Epoch')
plt.ylabel('Predicted Alpha')
plt.title('Convergence of Inferred Diffusivity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.semilogy(loss_history, 'g-', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
with torch.no_grad():
    u_pred_check = model(x_data, t_data).cpu().numpy()
plt.scatter(t_data.cpu().numpy(), u_data.cpu().numpy(), c='blue', s=5, alpha=0.4, label='Loaded Data')
plt.scatter(t_data_cpu.numpy(), u_pred_check, c='red', s=5, alpha=0.4, label='PINN Fit')
plt.xlabel('Time t')
plt.ylabel('Temperature u')
plt.title('Loaded Forward Data vs PINN')
plt.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
snapshot_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for idx, ti in enumerate(snapshot_times):
    x_test = torch.linspace(0, 1, 100, device=device).view(-1, 1)
    t_test = torch.ones(100, 1, device=device) * ti
    
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy().flatten()
    
    ax = axes[idx // 3, idx % 3]
    ax.plot(x_test.cpu().numpy(), u_pred, 'r-', linewidth=2, label='PINN Reconstruction')
    ax.set_title(f't = {ti:.1f}, alpha_pred = {alpha_pred.item():.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    if idx == 0: ax.legend()

plt.suptitle('Reconstructed Temperature Field from Loaded Data', fontsize=14)
plt.tight_layout()
plt.show()