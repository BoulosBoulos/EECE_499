# Full State Schema

Per System_Architecture.txt. State vector: `s_t = [s_ego, s_geom, s_vis, f^1, ..., f^N]`.

## 2.1 Ego state (6)

`[v, a, psi_dot, psi, a - a_prev, psi_dot - psi_dot_prev]`

## 2.2 Route & intersection geometry (12)

`[d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane, g_turn(3), rho(2)]`

- `d_stop`, `d_cz`, `d_exit`: distances along path
- `kappa`: curvature, `e_y`/`e_psi`: path errors
- `g_turn`: one-hot left/straight/right
- `rho`: right-of-way context

## 2.3 Visibility (6)

`[alpha_cz, alpha_cross, d_occ, dt_seen, sigma_percep, n_occ]`

## 2.4 Per-agent features (22 × top_N)

For each top-N agent: `[delta_x, delta_y, delta_vx, delta_vy, delta_psi, v_i, a_i, d_cz_i, d_exit_i, tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i, chi, pi_row, nu, sigma, type_onehot(3), mask]`

## Total dim

6 + 12 + 6 + top_N × 22 = 134 (top_N=5)
