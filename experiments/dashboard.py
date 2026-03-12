"""Streamlit dashboard for DRPPO training, ablation, and sensitivity analysis.

Run:  streamlit run experiments/dashboard.py --server.port 8501
"""

from __future__ import annotations

import os
import glob
import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise SystemExit("Install: pip install streamlit plotly pandas")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def _find_csvs(pattern: str, root: str = RESULTS_DIR) -> list[str]:
    return sorted(glob.glob(os.path.join(root, pattern)) +
                  glob.glob(os.path.join(root, "**", pattern), recursive=True))


def _load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


st.set_page_config(page_title="DRPPO Dashboard", layout="wide", page_icon="🚗")
st.title("DRPPO: Physics-Informed RL Dashboard")

tab_overview, tab_train, tab_ablation, tab_sensitivity, tab_violations, tab_intent, tab_tables = st.tabs([
    "Project Overview", "Training Curves", "Ablation Summary", "Sensitivity",
    "Violations", "Intent Model", "Raw Tables",
])

# ── Tab 0: Project Overview ─────────────────────────────────────────────────

with tab_overview:
    st.header("Project Summary")
    st.markdown("""
    **DRPPO** (Deep Recurrent Proximal Policy Optimization) is a physics-informed
    reinforcement learning framework for **autonomous driving at unsignalized
    T-intersections**. The ego vehicle must negotiate right-of-way with diverse
    road users (cars, motorcycles, pedestrians) under partial observability.

    **Simulation:** Each episode is run in [SUMO](https://eclipse.dev/sumo/)
    via TraCI. The ego approaches from the stem road and turns right onto the
    bar road, while other agents are spawned with randomized maneuvers, driving
    styles, and timing drawn from a per-episode behavior sampler.

    **Key idea:** Standard PPO is augmented with three physics-informed residual
    losses (TTC, stopping-distance, friction-circle) that can be applied to the
    **critic** (Design A), the **actor** (Design B), or **both**. A runtime
    **safety filter** can also override actions when physics constraints are
    violated. All of these are independently togglable ("plug-and-play").
    """)

    st.subheader("Scenarios")
    scenario_df = pd.DataFrame({
        "Scenario": ["1a", "1b", "1c", "1d", "2", "3", "4"],
        "Car": ["Yes", "Yes", "No", "No", "Yes", "Yes", "Yes"],
        "Pedestrian": ["No", "Yes", "No", "Yes", "No", "Yes", "Yes"],
        "Motorcycle": ["No", "No", "Yes", "Yes", "No", "No", "Yes"],
        "Pothole": ["No", "No", "No", "No", "Yes", "No", "Yes"],
        "Description": [
            "Ego + car only",
            "Ego + car + pedestrian",
            "Ego + motorcycle only",
            "Ego + motorcycle + pedestrian",
            "Ego + car + pothole",
            "Ego + car + pedestrian (complex)",
            "Ego + car + ped + moto + pothole (full)",
        ],
    })
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # ── State Vector ────────────────────────────────────────────────────────
    st.subheader("State Vector Definition")
    st.markdown("The observation `s_t` is a flat vector built from 4 groups:")
    state_df = pd.DataFrame([
        {"Group": "s_ego (6D)", "Index": "0-5",
         "Components": "v, a, psi_dot, psi, jerk (a-a_prev), yaw_accel (psi_dot-psi_dot_prev)"},
        {"Group": "s_geom (12D)", "Index": "6-17",
         "Components": "d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane, g_turn[3], rho[2]"},
        {"Group": "s_vis (6D)", "Index": "18-23",
         "Components": "alpha_cz, alpha_cross, d_occ, dt_seen, sigma_percep, n_occ"},
        {"Group": "f_agents (N x 22D)", "Index": "24-133",
         "Components": "Per top-5 agent: delta_xy[2], delta_v[2], delta_psi, v_i, a_i, d_cz_i, d_exit_i, tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i, chi_i, pi_row_i, nu_i, sigma_i, type_onehot[3], mask"},
    ])
    st.dataframe(state_df, use_container_width=True, hide_index=True)
    st.caption("Optional extras: +30D intent features (if `use_intent`), +1D pothole distance (if scenario has pothole).")

    # ── Variant Definitions ─────────────────────────────────────────────────
    st.subheader("Ablation Variants (10 Plug-and-Play Configurations)")
    variant_df = pd.DataFrame([
        {"Variant": "nopinn", "PINN Placement": "none", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Baseline PPO, no physics augmentation"},
        {"Variant": "pinn_critic", "PINN Placement": "critic", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Design A: physics loss on critic only"},
        {"Variant": "pinn_actor", "PINN Placement": "actor", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Design B: physics loss on actor only"},
        {"Variant": "pinn_both", "PINN Placement": "both", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Designs A+B: physics loss on both"},
        {"Variant": "pinn_ego", "PINN Placement": "critic", "L_ego": True,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Design A + ego dynamics prediction loss"},
        {"Variant": "pinn_no_ttc", "PINN Placement": "critic", "L_ego": False,
         "Safety Filter": False, "TTC": False, "Stop": True, "Fric": True,
         "Description": "Design A without TTC residual"},
        {"Variant": "pinn_no_stop", "PINN Placement": "critic", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": False, "Fric": True,
         "Description": "Design A without stopping-distance residual"},
        {"Variant": "pinn_no_fric", "PINN Placement": "critic", "L_ego": False,
         "Safety Filter": False, "TTC": True, "Stop": True, "Fric": False,
         "Description": "Design A without friction-circle residual"},
        {"Variant": "safety_filter", "PINN Placement": "none", "L_ego": False,
         "Safety Filter": True, "TTC": True, "Stop": True, "Fric": True,
         "Description": "No physics loss; action override at inference"},
        {"Variant": "pinn_critic_sf", "PINN Placement": "critic", "L_ego": False,
         "Safety Filter": True, "TTC": True, "Stop": True, "Fric": True,
         "Description": "Design A + safety filter"},
    ])
    st.dataframe(variant_df, use_container_width=True, hide_index=True)

    # ── Combination Counts ──────────────────────────────────────────────────
    st.subheader("Experiment Combination Counts")
    n_scenarios = 7
    n_variants = 10
    n_seeds = 5
    n_lambda = 7
    n_pinn_variants = 8
    n_none_variants = 2
    n_total_single_lambda = n_scenarios * n_variants * n_seeds
    n_total_sweep = n_scenarios * (n_pinn_variants * n_lambda + n_none_variants * 1) * n_seeds

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Scenarios", n_scenarios)
    col_b.metric("Variants", n_variants)
    col_c.metric("Seeds", n_seeds)

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Default ablation jobs", f"{n_total_single_lambda:,}")
    col_e.metric("Full sweep jobs (7 lambdas)", f"{n_total_sweep:,}")
    col_f.metric("Eval modes", "2 (det + stoch)")

    st.markdown(f"""
    - **Default ablation** (`make ablation`): {n_scenarios} scenarios x {n_variants} variants x {n_seeds} seeds = **{n_total_single_lambda:,}** training runs
    - **Full sensitivity sweep** (`make ablation-full`): {n_pinn_variants} PINN variants x {n_lambda} lambdas + {n_none_variants} baseline = **{n_total_sweep:,}** runs per scenario x {n_scenarios} scenarios = **{n_total_sweep:,}** total
    - **16-GPU parallel**: job manifest splits any configuration across GPUs with `make ablation-16gpu`
    """)

    # ── Architecture Workflow ───────────────────────────────────────────────
    st.subheader("Architecture Workflow")
    st.markdown("Interactive view of the DRPPO training and evaluation pipeline:")

    with st.expander("Training Pipeline", expanded=True):
        st.markdown("""
        ```
        SUMO Environment
            |
            v
        reset() --> obs_0, info_0 (pre-step)
            |
            v
        +--------------------------+
        | collect_rollouts()       |
        |   for t = 0..n_steps:    |
        |     store pre_step_info  |
        |     h_t = GRU hidden     |
        |     a_t ~ policy(obs_t)  |  <-- safety filter may override
        |     obs_{t+1}, r_t, info |
        |     store post_step_info |
        +--------------------------+
            |
            v
        Build PINN 'extra' dict from PRE-STEP info (aligned with s_t):
          d_cz, v, kappa, a_lon, ttc_min, ego transitions
            |
            v
        +--------------------------+
        | PPO train_step()         |
        |   actor_loss (clipped)   |
        |   vf_loss (MSE)          |
        |   + L_physics (critic)   |  <-- Design A (if pinn_placement = critic/both)
        |   + L_actor_phys (actor) |  <-- Design B (if pinn_placement = actor/both)
        |   + L_ego (dynamics err) |  <-- if use_l_ego = True
        |   entropy bonus          |
        +--------------------------+
            |
            v
        Logged metrics --> CSV (incremental)
        ```
        """)

    with st.expander("Physics Residuals", expanded=False):
        st.markdown(r"""
        **Three physics-informed residuals** (each independently togglable):

        | Residual | Formula | What it penalizes |
        |----------|---------|-------------------|
        | TTC | $\text{ReLU}(\tau_{thr} - \text{TTC}_{min})$ | Being too close in time to a conflict |
        | Stop | $\text{ReLU}(d_{stop}(v) - d_{cz})$ | Not having enough distance to stop |
        | Friction | $\text{ReLU}(a_{lat}^2 + a_{lon}^2 - (\mu g)^2)$ | Exceeding friction circle |

        **Design A (Critic):** $\mathcal{L}_{critic} = \mathbb{E}[V(s) \cdot \text{violation}(s)]$

        **Design B (Actor):** $\mathcal{L}_{actor} = \mathbb{E}[\text{violation}(s)^{\text{detach}} \cdot (-\log\pi(a|s))]$

        **Safety Filter:** At inference, if $d_{cz} < d_{stop}(v)$ or friction is violated, override action to STOP.
        """)

    with st.expander("Evaluation Pipeline", expanded=False):
        st.markdown("""
        ```
        Load checkpoint --> DRPPO (with same pinn_placement + safety_filter)
            |
            v
        For each episode:
          reset env, reset GRU hidden
          run 500 steps (deterministic or stochastic)
          record: return, collisions, TTC, pothole hits
            |
            v
        Aggregate: mean_return, collision_rate, mean_ttc, min_ttc
            |
            v
        Write to ablation_results.csv (with eval_mode column)
        ```
        """)

    with st.expander("16-GPU Parallel Pipeline", expanded=False):
        st.markdown("""
        ```
        Step 1: generate_jobs.py
          --> job_manifest.json (scenario x variant x lambda x seed)

        Step 2: launch_parallel_16gpu.sh
          --> 16 workers, each with CUDA_VISIBLE_DEVICES=i
          --> run_single_job.py --worker_index i --num_workers 16
          --> per-job CSV: jobs/eval_000001.csv, jobs/train_000001.csv

        Step 3: aggregate_results.py
          --> merge all jobs/ CSVs into ablation_results.csv + train_log.csv
          --> dashboard and plot commands work unchanged
        ```
        """)

    with st.expander("Behavior Diversity", expanded=False):
        st.markdown("""
        Each episode samples independent behaviors for all actors:

        **Car** (4 maneuvers x 7 styles):
        - Maneuvers: straight_left_right, straight_right_left, turn_left, turn_right
        - Styles: nominal, aggressive, timid, distracted, erratic, drunk, rule_violating

        **Pedestrian** (2 maneuvers x 7 styles):
        - Maneuvers: cross_left_right, cross_right_left
        - Styles: normal_walk, running, slow_elderly, stop_midway, hesitant, distracted_slow, jaywalking_fast

        **Motorcycle** (3 maneuvers x 6 styles):
        - Maneuvers: straight_right_left, straight_left_right, turn_into_stem
        - Styles: nominal, aggressive_fast, cautious, late_brake, swerving, yield_to_ego

        **Pothole**: randomly positioned within the conflict zone each episode.
        """)


# ── Tab 1: Training Curves ──────────────────────────────────────────────────

with tab_train:
    st.header("Training Curves")
    train_csvs = _find_csvs("train_*.csv")
    if not train_csvs:
        st.warning("No training CSVs found. Run `make train` first.")
    else:
        selected = st.multiselect("Select runs", train_csvs,
                                  default=train_csvs[:4],
                                  format_func=lambda p: os.path.basename(p))
        if selected:
            frames = {}
            for path in selected:
                df = _load_csv(path)
                if df is not None:
                    name = os.path.basename(path).replace("train_", "").replace(".csv", "")
                    df["run"] = name
                    frames[name] = df
            if frames:
                combined = pd.concat(frames.values(), ignore_index=True)
                numeric_cols = ["step", "episode_return", "episode_len", "actor_loss",
                                "vf_loss", "collision_count", "collision_rate",
                                "mean_ttc", "min_ttc", "entropy",
                                "l_physics", "l_actor_physics", "l_ego",
                                "viol_ttc_rate", "viol_stop_rate", "viol_fric_rate",
                                "viol_ttc_mag", "viol_stop_mag", "viol_fric_mag"]
                combined = _safe_numeric(combined, numeric_cols)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(combined, x="step", y="episode_return", color="run",
                                  title="Episode Return")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.line(combined, x="step", y="collision_rate", color="run",
                                  title="Collision Rate")
                    st.plotly_chart(fig, use_container_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    fig = px.line(combined, x="step", y="actor_loss", color="run",
                                  title="Actor Loss")
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    fig = px.line(combined, x="step", y="vf_loss", color="run",
                                  title="Critic (VF) Loss")
                    st.plotly_chart(fig, use_container_width=True)

                col5, col6 = st.columns(2)
                with col5:
                    fig = px.line(combined, x="step", y="mean_ttc", color="run",
                                  title="Mean TTC")
                    st.plotly_chart(fig, use_container_width=True)
                with col6:
                    fig = px.line(combined, x="step", y="entropy", color="run",
                                  title="Policy Entropy")
                    st.plotly_chart(fig, use_container_width=True)

                has_physics = "l_physics" in combined.columns and combined["l_physics"].notna().any()
                has_actor_phys = "l_actor_physics" in combined.columns and combined["l_actor_physics"].notna().any()
                if has_physics or has_actor_phys:
                    st.subheader("Physics Losses")
                    col7, col8 = st.columns(2)
                    with col7:
                        if has_physics:
                            fig = px.line(combined, x="step", y="l_physics", color="run",
                                          title="L_physics (Critic PINN)")
                            st.plotly_chart(fig, use_container_width=True)
                    with col8:
                        if has_actor_phys:
                            fig = px.line(combined, x="step", y="l_actor_physics", color="run",
                                          title="L_actor_physics (Actor PINN)")
                            st.plotly_chart(fig, use_container_width=True)
                        elif "l_ego" in combined.columns:
                            fig = px.line(combined, x="step", y="l_ego", color="run",
                                          title="L_ego (Dynamics Error)")
                            st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Ablation Summary ─────────────────────────────────────────────────

with tab_ablation:
    st.header("Ablation Results")
    abl_csvs = _find_csvs("ablation_results.csv")
    if not abl_csvs:
        st.warning("No ablation results found. Run `make ablation` first.")
    else:
        abl_path = st.selectbox("Ablation CSV", abl_csvs,
                                format_func=lambda p: os.path.basename(p))
        df = _load_csv(abl_path)
        if df is not None:
            df = _safe_numeric(df, ["mean_return", "std_return", "collision_rate",
                                     "mean_ttc", "min_ttc", "lambda_phys", "seed"])

            # Filter by eval mode if present
            if "eval_mode" in df.columns:
                modes = sorted(df["eval_mode"].unique())
                sel_mode = st.selectbox("Eval mode", modes, index=0)
                df = df[df["eval_mode"] == sel_mode]

            scenarios = sorted(df["scenario"].unique()) if "scenario" in df.columns else ["all"]
            sel_scen = st.multiselect("Scenarios", scenarios, default=scenarios)
            df_filt = df[df["scenario"].isin(sel_scen)] if "scenario" in df.columns else df

            agg = df_filt.groupby("variant").agg(
                mean_return=("mean_return", "mean"),
                std_return=("mean_return", "std"),
                collision_rate=("collision_rate", "mean"),
                n_runs=("mean_return", "count"),
            ).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(agg, x="variant", y="mean_return", error_y="std_return",
                             title="Mean Return by Variant", color="variant")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(agg, x="variant", y="collision_rate",
                             title="Collision Rate by Variant", color="variant")
                st.plotly_chart(fig, use_container_width=True)

            if "mean_ttc" in df_filt.columns:
                agg_ttc = df_filt.groupby("variant")["mean_ttc"].mean().reset_index()
                fig = px.bar(agg_ttc, x="variant", y="mean_ttc",
                             title="Mean TTC by Variant", color="variant")
                st.plotly_chart(fig, use_container_width=True)

            if "scenario" in df_filt.columns:
                st.subheader("Per-Scenario Breakdown")
                fig = px.bar(df_filt, x="scenario", y="mean_return", color="variant",
                             barmode="group", title="Return by Scenario x Variant")
                st.plotly_chart(fig, use_container_width=True)

            # Plug-and-play comparison
            if "pinn_placement" in df_filt.columns:
                st.subheader("PINN Placement Comparison")
                placement_agg = df_filt.groupby("pinn_placement").agg(
                    mean_return=("mean_return", "mean"),
                    collision_rate=("collision_rate", "mean"),
                ).reset_index()
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.bar(placement_agg, x="pinn_placement", y="mean_return",
                                 title="Return by PINN Placement", color="pinn_placement")
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig = px.bar(placement_agg, x="pinn_placement", y="collision_rate",
                                 title="Collision Rate by PINN Placement", color="pinn_placement")
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Full Results Table")
            st.dataframe(df_filt, use_container_width=True)


# ── Tab 3: Sensitivity ──────────────────────────────────────────────────────

with tab_sensitivity:
    st.header("Hyperparameter Sensitivity")
    abl_csvs = _find_csvs("ablation_results.csv")
    if not abl_csvs:
        st.warning("No ablation results found.")
    else:
        abl_path = st.selectbox("Sensitivity CSV", abl_csvs,
                                format_func=lambda p: os.path.basename(p),
                                key="sens_csv")
        df = _load_csv(abl_path)
        if df is not None and "lambda_phys" in df.columns:
            df = _safe_numeric(df, ["mean_return", "collision_rate", "mean_ttc", "lambda_phys"])
            n_lambdas = df["lambda_phys"].nunique()
            if n_lambdas > 1:
                agg = df.groupby(["variant", "lambda_phys"]).agg(
                    mean_return=("mean_return", "mean"),
                    std_return=("mean_return", "std"),
                    collision_rate=("collision_rate", "mean"),
                ).reset_index()

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(agg, x="lambda_phys", y="mean_return", color="variant",
                                  markers=True, title="Return vs lambda_phys")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.line(agg, x="lambda_phys", y="collision_rate", color="variant",
                                  markers=True, title="Collision Rate vs lambda_phys")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Only one lambda value found. Run with --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0")
        elif df is not None:
            st.info("No lambda_phys column in CSV.")


# ── Tab 4: Violations ───────────────────────────────────────────────────────

with tab_violations:
    st.header("Physics Violation Statistics")

    train_csvs = _find_csvs("train_*.csv")
    abl_train = _find_csvs("ablation_train_log.csv")
    all_viol_csvs = train_csvs + abl_train
    if not all_viol_csvs:
        st.warning("No training logs with violation data found.")
    else:
        selected = st.multiselect("Select logs", all_viol_csvs,
                                  default=all_viol_csvs[:4],
                                  format_func=lambda p: os.path.basename(p),
                                  key="viol_select")
        frames = []
        for path in selected:
            df = _load_csv(path)
            if df is not None:
                viol_cols = ["viol_ttc_rate", "viol_stop_rate", "viol_fric_rate",
                             "viol_ttc_mag", "viol_stop_mag", "viol_fric_mag"]
                has_viol = any(c in df.columns for c in viol_cols)
                if has_viol:
                    if "run" not in df.columns:
                        if "variant" in df.columns and "seed" in df.columns:
                            df["run"] = df["variant"].astype(str) + "_s" + df["seed"].astype(str)
                        else:
                            name = os.path.basename(path).replace(".csv", "")
                            df["run"] = name
                    frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = _safe_numeric(combined, ["step", "viol_ttc_rate", "viol_stop_rate",
                                                 "viol_fric_rate", "viol_ttc_mag",
                                                 "viol_stop_mag", "viol_fric_mag"])
            run_col = "run" if "run" in combined.columns else "variant"

            col1, col2 = st.columns(2)
            with col1:
                if "viol_ttc_rate" in combined.columns:
                    fig = px.line(combined, x="step", y="viol_ttc_rate", color=run_col,
                                  title="TTC Violation Rate")
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "viol_stop_rate" in combined.columns:
                    fig = px.line(combined, x="step", y="viol_stop_rate", color=run_col,
                                  title="Stop-Distance Violation Rate")
                    st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if "viol_fric_rate" in combined.columns:
                    fig = px.line(combined, x="step", y="viol_fric_rate", color=run_col,
                                  title="Friction Violation Rate")
                    st.plotly_chart(fig, use_container_width=True)
            with col4:
                mag_cols = [c for c in ["viol_ttc_mag", "viol_stop_mag", "viol_fric_mag"]
                            if c in combined.columns]
                if mag_cols:
                    fig = go.Figure()
                    for mc in mag_cols:
                        fig.add_trace(go.Box(y=combined[mc].dropna(),
                                             name=mc.replace("viol_", "").replace("_mag", "")))
                    fig.update_layout(title="Violation Magnitudes (Box Plot)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violation data in selected logs.")


# ── Tab 5: Intent Model ─────────────────────────────────────────────────────

with tab_intent:
    st.header("Intent/Style LSTM Training")
    intent_csvs = _find_csvs("intent_train_log.csv")
    if not intent_csvs:
        st.info("No intent training logs found. Run `make train-intent` first.")
    else:
        intent_path = st.selectbox("Intent log", intent_csvs,
                                   format_func=lambda p: os.path.basename(p))
        df = _load_csv(intent_path)
        if df is not None:
            df = _safe_numeric(df, ["epoch", "train_loss", "val_loss",
                                     "val_intent_acc", "val_style_acc"])
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Loss"))
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val Loss"))
                fig.update_layout(title="Intent LSTM: Loss")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                if "val_intent_acc" in df.columns:
                    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_intent_acc"], name="Intent Acc"))
                if "val_style_acc" in df.columns:
                    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_style_acc"], name="Style Acc"))
                fig.update_layout(title="Intent LSTM: Accuracy")
                st.plotly_chart(fig, use_container_width=True)


# ── Tab 6: Raw Tables ───────────────────────────────────────────────────────

with tab_tables:
    st.header("Raw Data Browser")
    all_csvs = _find_csvs("*.csv")
    if not all_csvs:
        st.info("No CSVs found in results/.")
    else:
        csv_path = st.selectbox("CSV file", all_csvs,
                                format_func=lambda p: os.path.relpath(p, RESULTS_DIR))
        df = _load_csv(csv_path)
        if df is not None:
            st.write(f"**{len(df)} rows x {len(df.columns)} columns**")
            st.dataframe(df, use_container_width=True, height=500)
            csv_download = df.to_csv(index=False)
            st.download_button("Download CSV", csv_download, file_name=os.path.basename(csv_path))


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("About")
    st.markdown("""
    **DRPPO Dashboard**

    Physics-informed recurrent PPO for T-intersection driving.

    **Plug-and-play variants:**
    - nopinn / pinn_critic / pinn_actor / pinn_both
    - pinn_ego / pinn_no_ttc / pinn_no_stop / pinn_no_fric
    - safety_filter / pinn_critic_sf

    **Tabs:**
    - Training Curves: return, losses, collisions, TTC, violations
    - Ablation: compare all variants + PINN placement
    - Sensitivity: lambda_phys sweep
    - Violations: per-term physics constraint tracking
    - Intent: LSTM training metrics
    - Raw Tables: browse any CSV
    """)
    st.markdown("---")
    st.markdown(f"Results dir: `{RESULTS_DIR}`")
