# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:12:29 2026

@author: LenovoTest
"""

"""

The thermal diffusivity alpha is treated as an additional NETWORK INPUT.
Consequently, one model learns a parametric solution u(x, t, alpha), rather
than learning a separate solution for only one fixed alpha. Training uses
alpha = 0.05, 0.1, and 0.2; evaluation also tests the unseen value alpha=0.15

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Select CUDA (GPU) if available and otherwise use the CPU. The device choice
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Problem definition and training hyperparameters
# Spatial domain: 0 <= x <= L. Time domain: 0 <= t <= T_end.
L = 1.0
T_end = 1.0


# The model sees solutions for these three diffusivities during training.
TRAIN_ALPHAS = [0.05, 0.1, 0.2]

# In order to test if the PINN can deal with untreated situation
TEST_ALPHA = 0.15  

# EPOCHS: number of gradient-descent updates.
EPOCHS = 5000
# LR: initial learning rate for the Adam optimizer. Learning rate: How fast the parameter change with the gradient
LR = 1e-3
# N_PDE: random PDE collocation points sampled for each update.
N_PDE = 3000
# N_BC / N_IC: boundary/initial points generated per training alpha.
N_BC = 200
N_IC = 200
# N_DATA: supervised solution samples selected per training alpha.
N_DATA = 300

# Weights for the individual terms in the total loss, setting all weights to 1 at first
LAMBDA_PDE = 1.0
LAMBDA_BC = 1.0
LAMBDA_IC = 1.0
LAMBDA_DATA = 1.0

# The solution of heat equation
def exact_solution(x, t, alpha):
    return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

# 2. Build the network
class ParametricPINN(nn.Module):
    # The network approximates a solution field by mapping input coordinates and physical parameters to one or more output values.
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, layers=3):
        # Initialize the parent nn.Module class before assigning any layers.
        # This enables PyTorch to register and manage the model parameters.
        super().__init__()
        # Create a list to store the network layers in their execution order.
        mods = [nn.Linear(in_dim, hidden_dim), ]
        # Add the remaining hidden layers.
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        # Add the final output layer. No activation function is applied here,
        mods += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*mods)
    
    def forward(self, *coords):
        # Evaluate the network for a batch of input samples
        # Returns: A tensor of shape [N, out_dim] containing the network predictions.
        return self.net(torch.cat(coords, dim=1))

# 3. Generate supervised, boundary, and initial training data
# Three groups of points are constructed for every alpha in TRAIN_ALPHAS: 
# Interior observation; Boundary points at x=0 and x=L, where u=0; Initial points at t=0, where u=sin(pi*x)
def generate_training_data():
    # A dictionary whose values are float32 tensors on `device`
    data_dict = {}

    # Lists temporarily collect scalar samples
    data_x, data_t, data_a, data_u = [], [], [], []

    # Observation data from the exact solution 
    for a in TRAIN_ALPHAS:
        # A tensor-product grid: 50 x coordinates and 30 t coordinates
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
    
    # For each alpha, N_BC//2 points lie at x=0 and N_BC//2 at x=L.
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
    
    # At t=0, randomly sample x in [0,1). The exact initial profile supplies the target sin(pi*x).
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

# 4. Define the physics-informed loss function
# Evaluate the total training loss for one selected PDE diffusivity
def compute_loss(model, data_dict, alpha_pde):
    # Randomly sample spatial and temporal coordinates for the PDE collocation points
    x_f = torch.rand(N_PDE, 1, device=device, requires_grad=True)
    t_f = torch.rand(N_PDE, 1, device=device, requires_grad=True)
    # Assign the selected diffusivity alpha_pde to every collocation point
    a_f = torch.ones(N_PDE, 1, device=device) * alpha_pde
    
    # Evaluate u_theta(x,t,alpha) at all collocation points in one batch
    u_f = model(x_f, t_f, a_f)

    # Compute the derivative of network
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

    # If the PDE is satisfied exactly, residual = u_t - alpha*u_xx = 0
    residual = u_t - a_f * u_xx
    loss_pde = torch.mean(residual**2)
    
    # Evaluate all stored boundary points and compare predictions with u=0 (No automatic differentiation is needed)
    u_bc_pred = model(data_dict['bc_x'], data_dict['bc_t'], data_dict['bc_a'])
    loss_bc = torch.mean((u_bc_pred - data_dict['bc_u'])**2)
    
    # Enforce u(x,0,alpha)=sin(pi*x) for every training alpha
    u_ic_pred = model(data_dict['ic_x'], data_dict['ic_t'], data_dict['ic_a'])
    loss_ic = torch.mean((u_ic_pred - data_dict['ic_u'])**2)
    
    # Fit the sparse exact-solution samples
    u_data_pred = model(data_dict['x'], data_dict['t'], data_dict['a'])
    loss_data = torch.mean((u_data_pred - data_dict['u'])**2)
    
    # Compute the total loss
    total_loss = LAMBDA_PDE * loss_pde + LAMBDA_BC * loss_bc + LAMBDA_IC * loss_ic + LAMBDA_DATA * loss_data
    
    return total_loss, loss_pde, loss_bc, loss_ic, loss_data

# 5. Train the parametric PINN
# Build and train the model, then return the trained model
def train():
    model = ParametricPINN(hidden_dim=64).to(device)
    # The optimizer is the algorithm updating the trainable parameter of the model according to the gradient of loss function
    # Adam adapts parameter updates using gradient moments. model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Make learning rate decrease with training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # Generate fixed data once before training to avoid regenerating supervised
    data_dict = generate_training_data()
    
    print("Start training...")
    print(f"Training alpha set: {TRAIN_ALPHAS}")
    print(f"Testing alpha: {TEST_ALPHA}")
    
    for epoch in range(EPOCHS):
        # Reset gradients because PyTorch accumulates them across backward calls.
        optimizer.zero_grad()
        
        # Randomly choose which alpha governs this epoch's fresh PDE points
        alpha_pde = np.random.choice(TRAIN_ALPHAS)
        
        # Forward pass plus all four loss evaluations
        total_loss, loss_pde, loss_bc, loss_ic, loss_data = compute_loss(
            model, data_dict, alpha_pde
        )

        # compute gradients of total_loss for all model parameters
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # .item() extracts Python scalars for printing without retaining computational graphs
        # Monitoring separate losses is more informative than total loss alone
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | Total: {total_loss.item():.3e} | "
                  f"PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | "
                  f"IC: {loss_ic.item():.3e} | Data: {loss_data.item():.3e} | "
                  f"PDE_alpha: {alpha_pde:.2f}")
    
    return model

# 6. Evaluate predictions as a time-evolution sequence
# Time evolution sequuence image
def plot_time_evolution(model, alpha_test, title_suffix=""):
    # Set evaluation mode. This network has no dropout or batch normalization,
    # so behavior is effectively unchanged here, but model.eval() is good habit
    model.eval()
    
    # Different time as different situation
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # x as x-axis
    x_test = np.linspace(0, L, 100)

    # Make the subfigure 2 lines and 3 subfigures each line
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Flatten the figure to make the later epoches convenient
    axes = axes.flatten()
    
    max_error = 0

    # Get the serial number and the value of time
    for idx, ti in enumerate(times):
        # Construct the input of network with 100 samples, each sample has different x but same t, alpha
        x_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)
        t_tensor = torch.ones(100, 1, device=device) * ti
        a_tensor = torch.ones(100, 1, device=device) * alpha_test

        # Inference does not need a computational graph. no_grad() saves memory
        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor, a_tensor).cpu().numpy().flatten()
        
        u_true = exact_solution(x_test, ti, alpha_test)

        # Absolute error is measured point by point.
        error = np.abs(u_pred - u_true)
        max_error = max(max_error, error.max())
        # relative L2 error is to measure the error of the whole curve
        l2_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        
        ax = axes[idx]
        ax.plot(x_test, u_true, 'b-', linewidth=2, label='Exact (Analytical)')
        ax.plot(x_test, u_pred, 'r--', linewidth=2, label='PINN Prediction')
        # Fill the translucent gray region between two curves to make difference more obvious
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

# 7. The main guard prevents training and file generation from running automatically
# if this file is imported as a module from another Python script
if __name__ == '__main__':
    model = train()

    # Verification 1 uses an alpha that appeared explicitly during training
    print("\n" + "="*50)
    print("Verification 1: Training alpha = 0.1")
    plot_time_evolution(model, alpha_test=0.1, title_suffix="[In-distribution]")

    # Verification 2 uses an unseen alpha between two training values.
    # Good accuracy here would suggest that the model learned a continuous dependence on alpha, rather than memorizing only the three training solutions
    print("\n" + "="*50)
    print(f"Verification 2：Test alpha (Untrained) = {TEST_ALPHA}")
    plot_time_evolution(model, alpha_test=TEST_ALPHA, title_suffix="[Out-of-distribution]")

    # Verification 3 checks another trained value at the lower boundary of the training parameter set.
    print("\n" + "="*50)
    print("Verification 3：Training alpha = 0.05")
    plot_time_evolution(model, alpha_test=0.05, title_suffix="[Boundary of training set]") 

    # 8. Generate noisy forward data for the inverse-PINN example
    alpha_save = 0.1   
    nx, nt = 100, 100
    
    x_grid = np.linspace(0, L, nx)
    t_grid = np.linspace(0, T_end, nt)
    X_save, T_save = np.meshgrid(x_grid, t_grid)

    # Flatten the [100,100] grids into 10,000 samples, each shaped as a row with one feature after view(-1,1)
    x_tensor = torch.tensor(X_save.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(T_save.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    a_tensor = torch.ones(nx * nt, 1, device=device) * alpha_save
    
    model.eval()
    with torch.no_grad():
        U_save = model(x_tensor, t_tensor, a_tensor).cpu().numpy()

    # Add independent Gaussian noise with standard deviation equal to 1% of the predicted field's standard deviation
    noise = 0.01 * np.std(U_save) * np.random.randn(*U_save.shape)
    U_save = U_save + noise

    # Save a single compressed-by-NumPy ZIP-style NPZ archive with named arrays.
    # x, t, and u are flattened to [10000]; true_alpha is a scalar array
    np.savez('forward_data.npz',
             x=X_save.flatten(),
             t=T_save.flatten(),
             u=U_save.flatten(),
             true_alpha=alpha_save)
    
    print(f"Saved {len(X_save.flatten())} points for alpha = {alpha_save} to forward_data.npz") 