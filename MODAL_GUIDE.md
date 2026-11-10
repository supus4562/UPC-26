# UPC '26 Problem B: Cloud GPU Simulation Guide (Modal.com)

This guide walks you through executing the **Massively Parallel Aerodynamic Inversion & Uncertainty Quantification Engine** on **Modal Cloud GPUs (H100 / H200 / RTX 6000 Ada / Blackwell)**.

---

## 1. Quick Setup (One-time only)

Make sure you have `modal` installed on your machine and authenticate with your Modal account:

```bash
# 1. Install modal
pip install -r requirements.txt

# 2. Authenticate with your modal.com account (opens your browser)
modal setup
```

---

## 2. Launching the Simulation on Cloud GPU

You run everything with a single command from your project folder. **No heavy computation runs locally**—Modal deploys the PyTorch container to their data center, runs the batch RK4 integrator on an H100/H200 GPU, and pipes the output figures and metrics right back to your workspace.

### Standard Run (1,000,000 Monte Carlo trajectories, 5mm camera noise):
```bash
modal run modal_simulation.py
```

### Ultra-Scale Run (5,000,000 or 10,000,000 trajectories):
Because you have industrial Blackwell / H200 / H100 compute, you can push the sample size to millions with sub-second GPU runtimes:
```bash
# 5 Million samples, 3mm camera jitter
modal run modal_simulation.py --samples 5000000 --noise-mm 3.0
```

---

## 3. How the Modal Wrapper Works

Here is what happens behind the scenes:
1. **Container Definition (`sim_image`)**:
   Modal automatically packages Debian Linux, Python 3.11, PyTorch 2.2+, NumPy, SciPy, and Matplotlib into a serverless cloud image.
2. **GPU Dispatch (`@app.function(gpu="H100")`)**:
   Modal spins up a dedicated GPU worker instance on their cloud cluster in ~2 seconds.
3. **Massive Parallel Integration (`simulate_trajectories_batch`)**:
   The 3D state equations:
   $$\frac{d\vec{r}}{dt} = \vec{v}, \quad \frac{d\vec{v}}{dt} = \vec{g} - \gamma \|\vec{v}\|\vec{v} + \vec{k}_M \times \vec{v}$$
   are integrated for all $1,000,000+$ trajectories simultaneously as a 2D GPU Tensor.
4. **Local Download (`@app.local_entrypoint`)**:
   Once the GPU finishes, the 300-DPI publication figures and the JSON metrics file are streamed directly to your local `./simulation_output/` folder.

---

## 4. Selecting Different GPUs in Modal

In [`modal_simulation.py`](file:///home/supus/Projects/UPC%20%2726/modal_simulation.py), line 42 specifies the target GPU:
```python
@app.function(
    gpu="H100",  # Options: "H100", "A100", "A10G", "L4", or "any"
    image=sim_image,
    ...
)
```
* If your organization has specific H200 / B200 / RTX 6000 access under custom tags in Modal, you can set `gpu="H100"` or configure your Modal workspace GPU flag.
* If you want Modal to pick whatever high-tier GPU is instantly available with zero queue time, use `gpu="any"`.

---

## 5. Generated Artifacts for Your UPC Paper

Once the run finishes, the following files will be ready in `./simulation_output/`:

| File | Description for Paper |
| :--- | :--- |
| `trajectory_3d.png` | **Figure 1**: High-resolution 3D flight trajectory from the penalty spot to the goal plane, showing initial tracking points $(t_0, t_1, t_2)$ and the 3D goal frame. |
| `goal_plane_uncertainty.png` | **Figure 2**: 2D goal face $(7.32\text{m} \times 2.44\text{m})$ showing the Monte Carlo density cloud, nominal impact point, analytical $95\%$ confidence ellipse, and the goalkeeper's dynamic reach envelope. |
| `noise_scaling.png` | **Figure 3**: Error amplification scaling curve showing how camera measurement uncertainty $\sigma_{\text{sensor}}$ scales to goal impact uncertainty $(\sigma_y, \sigma_z)$. |
| `environmental_sensitivity.png` | **Figure 4**: Sensitivity plots showing trajectory drift due to crosswinds and high-altitude air density reduction (e.g. Mexico City / La Paz). |
| `summary_metrics.json` | Exact numerical values (nominal velocity, drag parameter $\gamma$, Magnus vector $\vec{k}_M$, arrival coordinates, $95\%$ ellipse semi-axes, goalkeeper save probability). |
