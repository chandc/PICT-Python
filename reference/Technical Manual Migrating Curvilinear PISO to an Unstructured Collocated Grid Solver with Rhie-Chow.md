## Technical Manual: Migrating Curvilinear PISO to an Unstructured Collocated Grid Solver with Rhie-Chow

This manual serves as a definitive architectural blueprint for migrating a structured, curvilinear PISO fluid solver into a fully unstructured, collocated Finite Volume Method (FVM) solver. It is tailored for high-performance deployment adhering to the canonical hydrodynamic benchmark standards used in frameworks like **HydroGym** (e.g., $Re = 100$ 2D circular cylinder flow control).

------

## 1. Core Platform Context & Benchmark Specifications

When validating your unstructured solver against the standard **HydroGym 2D circular cylinder benchmark**, the configuration must duplicate the following canonical specifications:

## 1.1 HydroGym 2D Cylinder Environment Specs

- **Domain Size:** Extends $51.2 D$ in the streamwise direction ($x$-axis) and $48 D$ in the cross-stream direction ($y$-axis), with the cylinder centered at the origin $(0,0)$. The inlet boundary sits at $x \approx -16 D$ and the outlet boundary at $x \approx +35.2 D$.
- **Boundary Conditions:**
  - *Inlet:* Uniform Dirichlet velocity, $\mathbf{u} = (U_\infty, 0) = (1, 0)$.
  - *Outlet:* Pressure outflow set to zero reference pressure, $p = 0$.
  - *Top/Bottom:* Free-slip / symmetry boundary conditions ($u_y = 0$, $\frac{\partial u_x}{\partial y} = 0$).
  - *Cylinder Wall:* No-slip conditions adjusted dynamically via continuous Dirichlet values mapping either to a single rotary actuation scalar ($\Omega$) or localized parabolic synthetic jets.
- **Standard Uncontrolled Baseline Statistics ($Re = 100$):**
  - Strouhal Number ($St$): $0.165 \pm 0.003$ (Vortex shedding frequency).
  - Mean Drag Coefficient ($\overline{C}_d$): $1.34 - 1.35$.
  - Root-Mean-Square Lift ($C_l\text{ rms}$): $0.22 - 0.24$.
- **Active Flow Control Optimization Reward / Loss Function:**
  $$L = \sum_{t=1}^{T} \left( C_d(t) + \alpha \vert{}a_t\vert{} + \omega \vert{}C_l(t)\vert{} \right)$$ 
  Where $a_t$ represents the continuous control effort (actuator amplitude), and $\alpha, \omega$ are penalization weights.

------

## 2. Mathematical Discretization & Data Structure Transformations

## 2.1 From Structured Indices to Flat Arrays

Curvilinear grids rely on multi-dimensional coordinate tensors where neighboring cell locations are discovered implicitly via index shifts (e.g., $U[i+1][j]$). Unstructured grids discard index-based topologies entirely. The mesh is flattened into 1D arrays, and connectivity is managed using explicit index mapping arrays.

## Computational Grid Topology Layout

```unset
        Cell P (Index: c_own)            Face f (Index: f)          Cell N (Index: c_nei)
     ┌────────────────────────┐                  │                 ┌────────────────────────┐
     │                        │                  │                 │                        │
     │           •            ├──────────────────┼─────────────────┤           •            │
     │        Centroid        │                  │                 │        Centroid        │
     │      x_P (c_own)       │                  │                 │      x_N (c_nei)       │
     │                        │                  │                 │                        │
     └────────────────────────┘                  │                 └────────────────────────┘
                                                 └──► Normal Vector S_f = A_f * n_f
```

To vectorize execution in Python and avoid performance-killing loops, use these flat array dimensions:

- **Cell Arrays:** $\text{shape}=(N_{\text{cells}},)$ stores cellular scalar fields ($p$, $\rho$) and cell volumes ($V_C$).
- **Face Arrays:** $\text{shape}=(N_{\text{faces}},)$ stores face fluxes ($U_n = \mathbf{u} \cdot \mathbf{n}$), face areas ($A_f$), and normal vectors ($\mathbf{n}_f$).
- **Connectivity Tables:** `face_to_cells`: A $(N_{\text{faces}}, 2)$ array where $[f, 0]$ maps the Owner cell ID ($P$) and $[f, 1]$ maps the Neighbor cell ID ($N$).

## 2.2 Triangular Simplex Geometric Formulation

In 2D unstructured meshes, the fundamental geometric cell is the triangular simplex. Let a triangular cell $C$ be defined by three vertices ordered counter-clockwise: $\mathbf{x}_1 = (x_1, y_1)$, $\mathbf{x}_2 = (x_2, y_2)$, and $\mathbf{x}_3 = (x_3, y_3)$.

## Volume (Area) Calculation

The exact volume $V_C$ of the triangular simplex is computed using a 3x3 coordinate matrix determinant:
$$V_C = \frac{1}{2} \left\vert{} \det \begin{bmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{bmatrix} \right\vert{} = \frac{1}{2} \Big\vert{} x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2) \Big\vert{}$$ 

## Cell Centroid

Scalar storage and collocated velocity tracking are assigned to the geometric centroid $\mathbf{x}_C$:
$$\mathbf{x}_C = \frac{1}{3} (\mathbf{x}_1 + \mathbf{x}_2 + \mathbf{x}_3)$$ 

## Face Area Vectors

Each bounding face $f$ possesses an outward face area vector $\mathbf{S}_f = A_f \mathbf{n}_f$. For a face bounded by vertices $\mathbf{a} = (x_a, y_a)$ and $\mathbf{b} = (x_b, y_b)$, the components are evaluated as:
$$\mathbf{S}_f = \begin{bmatrix} S_{f,x} \\ S_{f,y} \end{bmatrix} = \begin{bmatrix} y_b - y_a \\ -(x_b - x_a) \end{bmatrix}$$ 

## 2.3 Spatial Derivatives via Finite Volume Method

Curvilinear transformation metrics are abandoned. The spatial gradient of a scalar field $\phi$ at the cell center is computed using the **Gauss Divergence Theorem** over the bounding faces $f$ enclosing cell $C$:
$$\nabla \phi_C = \frac{1}{V_C} \sum_{f \in \text{faces}(C)} \phi_f \mathbf{S}_f$$ 

To guarantee geometric conservation, the grid must satisfy the **Geometric Closure Constraint** for every cell, balancing to machine precision:
$$\sum_{f \in \text{faces}(C)} \mathbf{S}_f = \mathbf{0}$$ 

------

## 3. The Core Purpose of Least-Squares Gradient Reconstruction

In an unstructured collocated grid solver, the **Least-Squares (LSQ) gradient reconstruction** serves four critical architectural functions:

1. **Driving Fluid Acceleration:** In the Navier-Stokes equations, momentum is physically driven by the pressure gradient force ($-\nabla p$). Because pressure ($p$) is a single scalar stored at cell centers, the LSQ matrix acts as the multidimensional slope evaluation tool.
2. **Feeding the Rhie-Chow Interpolation:** Suppressing pressure-velocity checkerboard oscillations requires tracking the difference between the direct face gradient and the interpolated cell-centered gradients ($\overline{\nabla p}$). The LSQ reconstruction calculates the cell-centered fields that are averaged to build this stability correction.
3. **Absorbing Mesh Skewness (Non-Orthogonal Corrections):** On arbitrary triangular meshes, face calculations must be split into implicit orthogonal components and explicit non-orthogonal corrections: $\text{Flux}_{\text{Non-Orthogonal}} = \mathbf{T}_f \cdot \overline{(\nabla p)}_f$. The LSQ gradient yields the full gradient vectors that are dotted with the skewness vector $\mathbf{T}_f$.
4. **Preserving Spatial Accuracy Constraints:** Alternative methods (like Green-Gauss integration) degrade in numerical accuracy on highly stretched or distorted triangular elements. The LSQ method mathematically minimizes the root-mean-square error across all immediate spatial neighbors, **guaranteeing first-order spatial accuracy even on highly skewed grids.**

------

## 4. Geometric Corrections for Mesh Non-Orthogonality

When the vector $\mathbf{d}_f = \mathbf{x}_N - \mathbf{x}_P$ connecting adjacent cell centroids is not parallel to the face normal vector $\mathbf{n}_f$, evaluating gradients introduces interpolation errors. To maintain diagonal dominance in your linear matrices, the face area vector $\mathbf{S}_f$ must be split using the **Over-Relaxed Approach**:

```unset
                 Cell P                              Cell N
             ┌────────────┐                      ┌────────────┐
             │            │      Face f (S_f)    │            │
             │     • P ───┼──────────────────────┼───► • N    │
             │            │       /              │            │
             └────────────┘      / ◄── E_f       └────────────┘
                                /
                               / ◄─── T_f (Skewness/Non-Orthogonality)
```

$$\mathbf{S}_f = \mathbf{E}_f + \mathbf{T}_f$$ 

## 4.1 Orthogonal Vector ($\mathbf{E}_f$)

The vector component aligned directly along the cell-center-to-cell-center path:
$$\mathbf{E}_f = \frac{\mathbf{d}_f}{\mathbf{d}_f \cdot \mathbf{d}_f} (\mathbf{d}_f \cdot \mathbf{S}_f)$$ 

## 4.2 Non-Orthogonal Correction Vector ($\mathbf{T}_f$)

The explicit correction component that absorbs the grid skewness:
$$\mathbf{T}_f = \mathbf{S}_f - \mathbf{E}_f$$ 

------

## 5. Time-Step Independent Rhie-Chow Damping Formulation

On a collocated grid, the mass-conserving face flux velocity $U_f$ must be regularized to prevent pressure decoupling. In transient simulations, the momentum diagonal coefficient $\mathcal{A}_P$ shifts dynamically with your time step: $\mathcal{A}_P = \frac{\rho V_P}{\Delta t} + a_P^{\text{spatial}}$.

To prevent the stabilizing pressure-damping effect from vanishing as $\Delta t \to 0$, modern solvers isolate the purely spatial coefficients:

$$U_f = \overline{\mathbf{u}}_f \cdot \mathbf{n}_f - \overline{\left(\frac{V}{\mathcal{A}^{\text{spatial}}}\right)}_f \left[ (\nabla p)_f \cdot \mathbf{n}_f - \overline{(\nabla p)}_f \cdot \mathbf{n}_f \right]$$ 

Where:

- $\overline{\mathbf{u}}_f$: Linear interpolation of cell velocities to the face.
- $(\nabla p)_f \cdot \mathbf{n}_f$: The direct local face gradient flux including skewness adjustments:
  $$(\nabla p)_f \cdot \mathbf{n}_f \approx \frac{\mathbf{E}_f \cdot \mathbf{E}_f}{A_f \Vert\mathbf{d}_f\Vert} \left( \frac{p_N - p_P}{\Vert\mathbf{d}_f\Vert} \right) + \frac{\mathbf{T}_f \cdot \overline{(\nabla p)}_f}{A_f}$$ 
- $\overline{(\nabla p)}_f \cdot \mathbf{n}_f$: Linear interpolation of full cell-centered LSQ pressure gradients to the face.

------

## 6. Global Least-Squares System Matrix Assembly

Because the LSQ matrix operator equations depend entirely on the static mesh geometry, the matrix coefficients are assembled globally once during initialization.

For a target cell $C$, let $\mathbf{\delta x}_{CN} = \mathbf{x}_N - \mathbf{x}_C = \begin{bmatrix} \Delta x_{CN} & \Delta y_{CN} \end{bmatrix}^T$ be the displacement vector to neighbor cell $N$. We define the spatial mapping system across all surrounding neighbors $N \in \text{nb}(C)$ using the inverse square distance weights $w_{CN} = \frac{1}{\Vert\mathbf{\delta x}_{CN}\Vert^2}$:

$$\mathbf{M}_C \nabla p_C = \mathbf{b}_C$$ 

Where the normal tensor matrix $\mathbf{M}_C$ is a symmetric $2 \times 2$ accumulator computed as:
$$\mathbf{M}_C = \mathbf{A}_C^T \mathbf{W}_C \mathbf{A}_C = \begin{bmatrix} \sum w_{CN} \Delta x_{CN}^2 & \sum w_{CN} \Delta x_{CN} \Delta y_{CN} \\ \sum w_{CN} \Delta x_{CN} \Delta y_{CN} & \sum w_{CN} \Delta y_{CN}^2 \end{bmatrix}$$ 

The right-hand side source vector $\mathbf{b}_C$ aggregates the spatial pressure differences:
$$\mathbf{b}_C = \mathbf{A}_C^T \mathbf{W}_C \Delta \mathbf{p} = \begin{bmatrix} \sum w_{CN} \Delta x_{CN} (p_N - p_C) \\ \sum w_{CN} \Delta y_{CN} (p_N - p_C) \end{bmatrix}$$ 

The inversion of $\mathbf{M}_C$ is calculated analytically using the local determinant $\mathbb{D}_C = M_{11}M_{22} - M_{12}^2$:
$$\mathbf{M}_C^{-1} = \frac{1}{\mathbb{D}_C} \begin{bmatrix} \sum w_{CN} \Delta y_{CN}^2 & -\sum w_{CN} \Delta x_{CN} \Delta y_{CN} \\ -\sum w_{CN} \Delta x_{CN} \Delta y_{CN} & \sum w_{CN} \Delta x_{CN}^2 \end{bmatrix}$$ 

Multiplying $\mathbf{M}_C^{-1}$ by $\mathbf{b}_C$ yields a direct linear combination of surrounding pressures. These coefficients are mapped globally into two sparse operator matrices, $\mathbf{G}_x$ and $\mathbf{G}_y$, matching the total system size $(N_{\text{cells}}, N_{\text{cells}})$, allowing you to evaluate cell-centered gradient fields instantly:
$$\mathbf{\nabla p}_x = \mathbf{G}_x \cdot \mathbf{p}$$ 
$$\mathbf{\nabla p}_y = \mathbf{G}_y \cdot \mathbf{p}$$ 

------

## 7. Actuator Boundary Mapping (Zero-Net-Mass-Flux Synthetic Jets)

To handle continuous surface boundary actuation (such as HydroGym's upper and lower synthetic jet slots centered at $90^\circ$ and $270^\circ$), boundary zones are mapped using spatial profile scaling constraints.

```unset
                  Actuator Profile Injection (Dirichlet slot boundary)
                             Maximum Velocity (V_jet) at center
                                         ▲
                                        / \
                                       /   \
                                      /     \
                                    ─┴───────┴─
                                   Slot Boundaries (Velocity = 0)
```

For a boundary face $f_b$ located within the actuator slot array, let $s$ be the localized arc length distance measured from the geometric center of the slot patch, and let $R_{\text{half}}$ be the spatial half-width of the slot opening. The Dirichlet boundary velocity normal flux is applied as a spatial parabola:
$$U_{n, f_b} = V_{\text{jet}} \cdot \left[ 1.0 - \left(\frac{s}{R_{\text{half}}}\right)^2 \right]$$ 

------

## 8. Verification Protocol Suite

## 8.1 Geometric Sub-Component Verification

## Unit Test Case 1: Geometric Closure Constraint

- **Test Setup:** For every cell in your parsed mesh array, compute the vector sum of all bounding face area vectors $\mathbf{S}_f$.
- **Mathematical Success Criteria:**
  $$\max_{C} \left\Vert \sum_{f \in \text{faces}(C)} \mathbf{S}_f \right\Vert < 10^{-14}$$ 
- **Failure Mode:** If the maximum norm violates machine precision limits, your face normal vector orientation array contains directional errors.

## Unit Test Case 2: Least-Squares Linear Gradient Reconstruction Accuracy

- **Test Setup:** Initialize a known linear pressure distribution across the domain: $p(\mathbf{x}) = 3x - 2y$.
- **Mathematical Success Criteria:** Pipe the cell centroid coordinates into the LSQ gradient reconstruction function. The evaluated gradient must exactly yield:
  $$\nabla p_C = \begin{bmatrix} 3.0 & -2.0 \end{bmatrix} \quad \forall C, \quad \text{with } L_\infty \text{ error} < 10^{-12}$$ 

------

## 8.2 Overall Code System Verification Benchmarks

To confirm that your complete unstructured collocated PISO architecture achieves physical validation, execute the following benchmark timeline:

1. **Test Case Specification:** Initialize a 2D unconfined circular cylinder domain matching the HydroGym specs ($\text{Domain} = 51.2D \times 48D$, mesh elements resolved to the "coarse" standard $\approx 10,000$ cells, $Re = 100$).
2. **Transient Elimination Window:** Run the simulation with zero active boundary control for at least **150 to 200 non-dimensional convective time units ($t_c$)**. This gives the natural Hopf bifurcation ample time to break wake symmetry, amplify, and saturate into a clean limit-cycle vortex street.
3. **Measurement Sampling:** Sample the flow for an additional **50 to 100 $t_c$** (capturing roughly 8 to 16 full shedding cycles) to gather clean statistics.
4. **Target Hydrodynamic Metrics Validation:** Verify that your steady-state metrics fall within the strict validation bounds below:

| Verified System Metric           | Target Lower Bound | Target Upper Bound | Target Numerical Standard                    |
| :------------------------------- | :----------------- | :----------------- | :------------------------------------------- |
| **Strouhal Number ($St$)**       | 0.162              | 0.168              | **0.165** (Dominant frequency peak via FFT)  |
| **Mean Drag ($\overline{C}_d$)** | 1.33               | 1.36               | **1.35** (Time-average value over 10 cycles) |
| **Lift RMS ($C_l\text{ rms}$)**  | 0.21               | 0.25               | **0.23** (Root-mean-square amplitude limit)  |

------

Now that the entire structural reference, full standard LaTeX equations, and ASCII system layouts are compiled, would you like to build out the **PISO time-stepping predictor-corrector sequence matrices** next, or explore how to bundle this custom backend into a **Gymnasium environment wrapper class** for RL training?