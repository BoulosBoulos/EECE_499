"""Train the IntentStylePredictor LSTM ensemble (Phase 0F / v9).

Architecture: bidirectional 3-layer LSTM @ hidden=384 with a 1D CNN frontend.
Training: self-supervised reconstruction pretraining → class-balanced + label-
smoothed CE classification with cosine LR + AdamW, repeated for an ensemble
of 3 models with different seeds. Probability-averaged across members at eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


ENSEMBLE_SEEDS = [42, 1729, 7919]


def collect_intent_data(envs, n_episodes: int = 200, max_steps: int = 200,
                        window_cap: int = 50):
    """Collect (observation_history, intent_label, style_label) tuples from env(s).

    History buffer per agent capped at `window_cap` timesteps (default 50 for
    Phase 0F so 50-step windows are usable for both pretraining and inference).

    `envs` may be a single SumoEnv (legacy) or a list of SumoEnv instances.
    When a list is provided, episodes are round-robin assigned across envs.
    Seeds are still set to `ep` so each episode index → distinct seed within
    its scenario (across scenarios, the same seed produces different
    behaviour because the SumoEnv RNG is reset to ep on every reset()).
    """
    from state.builder import _rot2d, _wrap
    samples = []

    env_list = envs if isinstance(envs, (list, tuple)) else [envs]

    for ep in range(n_episodes):
        env = env_list[ep % len(env_list)]
        obs, info = env.reset(seed=ep)
        bcfg = info.get("behavior")
        agent_histories = {}

        for step in range(max_steps):
            action = np.random.randint(0, 5)
            obs, r, term, trunc, info = env.step(action)
            raw = info.get("raw_obs", {})
            ego = raw.get("ego", {})
            p_e = np.array(ego.get("p", [0, 0]))
            psi_e = float(ego.get("psi", 0))
            v_e = float(ego.get("v", 0))
            v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])

            for ag in raw.get("agents", []):
                aid = ag.get("id", "?")
                if aid not in agent_histories:
                    agent_histories[aid] = []
                p_i = np.array(ag["p"])
                psi_i = float(ag.get("psi", 0))
                v_i = float(ag.get("v", 0))
                d_cz = float(ag.get("d_cz", 1e6))
                v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])
                R = _rot2d(-psi_e)
                dp = p_i - p_e
                delta_xy = R @ dp
                delta_v = R @ (v_i_vec - v_e_vec)
                delta_psi = _wrap(psi_i - psi_e)
                t_cpa = np.clip(-np.dot(delta_xy, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
                p_cpa = delta_xy + t_cpa * delta_v
                d_cpa = np.linalg.norm(p_cpa)
                z = [delta_xy[0], delta_xy[1], delta_v[0], delta_v[1], delta_psi, d_cz, d_cpa,
                     ag.get("nu", 1.0), ag.get("sigma", 0.1),
                     ag.get("v", 0.0), ag.get("a", 0.0), ag.get("sigma_driver", 0.15)]
                agent_histories[aid].append(z)
                if len(agent_histories[aid]) > window_cap:
                    agent_histories[aid].pop(0)

            if term or trunc:
                break

        if bcfg is None:
            continue

        for aid, hist in agent_histories.items():
            if len(hist) < 5:
                continue
            h = np.array(hist, dtype=np.float32)
            if "other" in aid and bcfg.car:
                intent_label = bcfg.car_intent_label
                style_label = bcfg.car_style_label
            elif "ped" in aid and bcfg.pedestrian:
                intent_label = bcfg.ped_intent_label
                style_label = bcfg.ped_style_label
            elif "motorcyclist" in aid and bcfg.motorcycle:
                intent_label = bcfg.moto_intent_label
                style_label = bcfg.moto_style_label
            else:
                continue
            samples.append((h, intent_label, style_label))

        if (ep + 1) % 50 == 0:
            print(f"Collected {ep+1}/{n_episodes} episodes, {len(samples)} samples so far")

    return samples


def evaluate_per_class(model, samples, batch_size: int, device: str) -> dict:
    """Compute per-class accuracy on `samples` for both intent and style heads
    using a single model.
    """
    model.eval()
    intent_correct = [0, 0, 0]
    intent_total = [0, 0, 0]
    style_correct = [0, 0, 0]
    style_total = [0, 0, 0]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            if not batch:
                continue
            max_len = max(len(s[0]) for s in batch)
            x = np.zeros((len(batch), max_len, 12), dtype=np.float32)
            il_arr = np.zeros(len(batch), dtype=np.int64)
            sl_arr = np.zeros(len(batch), dtype=np.int64)
            for i, (h, il, sl) in enumerate(batch):
                x[i, :len(h)] = h
                il_arr[i] = il
                sl_arr[i] = sl
            x_t = torch.FloatTensor(x).to(device)
            ip, sp, _, _ = model(x_t, None)
            ip_pred = ip[:, -1, :].argmax(dim=-1).cpu().numpy()
            sp_pred = sp[:, -1, :].argmax(dim=-1).cpu().numpy()
            for i in range(len(batch)):
                il = int(il_arr[i])
                sl = int(sl_arr[i])
                intent_total[il] += 1
                style_total[sl] += 1
                if int(ip_pred[i]) == il:
                    intent_correct[il] += 1
                if int(sp_pred[i]) == sl:
                    style_correct[sl] += 1
    intent_acc = [intent_correct[c] / max(intent_total[c], 1) for c in range(3)]
    style_acc = [style_correct[c] / max(style_total[c], 1) for c in range(3)]
    overall_intent = sum(intent_correct) / max(sum(intent_total), 1)
    overall_style = sum(style_correct) / max(sum(style_total), 1)
    return {
        "intent": {"per_class_acc": intent_acc, "counts": intent_total, "overall": overall_intent},
        "style":  {"per_class_acc": style_acc, "counts": style_total, "overall": overall_style},
    }


def evaluate_ensemble(models, samples, batch_size: int, device: str) -> dict:
    """Per-class accuracy from an ensemble: mean probability across members."""
    for m in models:
        m.eval()
    intent_correct = [0, 0, 0]
    intent_total = [0, 0, 0]
    style_correct = [0, 0, 0]
    style_total = [0, 0, 0]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            if not batch:
                continue
            max_len = max(len(s[0]) for s in batch)
            x = np.zeros((len(batch), max_len, 12), dtype=np.float32)
            il_arr = np.zeros(len(batch), dtype=np.int64)
            sl_arr = np.zeros(len(batch), dtype=np.int64)
            for i, (h, il, sl) in enumerate(batch):
                x[i, :len(h)] = h
                il_arr[i] = il
                sl_arr[i] = sl
            x_t = torch.FloatTensor(x).to(device)
            ip_acc = None
            sp_acc = None
            for m in models:
                ip, sp, _, _ = m(x_t, None)
                if ip_acc is None:
                    ip_acc = ip[:, -1, :].clone()
                    sp_acc = sp[:, -1, :].clone()
                else:
                    ip_acc = ip_acc + ip[:, -1, :]
                    sp_acc = sp_acc + sp[:, -1, :]
            ip_avg = ip_acc / float(len(models))
            sp_avg = sp_acc / float(len(models))
            ip_pred = ip_avg.argmax(dim=-1).cpu().numpy()
            sp_pred = sp_avg.argmax(dim=-1).cpu().numpy()
            for i in range(len(batch)):
                il = int(il_arr[i])
                sl = int(sl_arr[i])
                intent_total[il] += 1
                style_total[sl] += 1
                if int(ip_pred[i]) == il:
                    intent_correct[il] += 1
                if int(sp_pred[i]) == sl:
                    style_correct[sl] += 1
    intent_acc = [intent_correct[c] / max(intent_total[c], 1) for c in range(3)]
    style_acc = [style_correct[c] / max(style_total[c], 1) for c in range(3)]
    overall_intent = sum(intent_correct) / max(sum(intent_total), 1)
    overall_style = sum(style_correct) / max(sum(style_total), 1)
    return {
        "intent": {"per_class_acc": intent_acc, "counts": intent_total, "overall": overall_intent},
        "style":  {"per_class_acc": style_acc, "counts": style_total, "overall": overall_style},
    }


def pretrain_lstm(model, samples, n_epochs: int = 30, batch_size: int = 64,
                  device: str = "cuda", window: int = 50, past_len: int = 40):
    """Self-supervised pretraining: predict the future `(window-past_len)` timesteps
    from the past `past_len` timesteps. Uses MSE on the recon_head output.
    Only samples with len >= window are used.
    """
    if n_epochs <= 0:
        return model
    full_traj_samples = [s for s in samples if len(s[0]) >= window]
    if not full_traj_samples:
        print(f"  [pretrain] no samples with len>={window}; skipping pretraining.")
        return model
    print(f"  [pretrain] {len(full_traj_samples)} eligible samples, "
          f"window={window}, past_len={past_len}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_future = window - past_len
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(full_traj_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(full_traj_samples), batch_size):
            batch = full_traj_samples[start:start + batch_size]
            if not batch:
                continue
            x = torch.FloatTensor(
                np.stack([s[0][:window] for s in batch])
            ).to(device)
            x_past = x[:, :past_len, :]
            x_future = x[:, past_len:window, :]

            x_in = model.conv_frontend(x_past.transpose(1, 2)).transpose(1, 2)
            out, _ = model.lstm(x_in)
            pred = model.recon_head(out[:, -n_future:, :])

            loss = nn.functional.mse_loss(pred, x_future)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"  [pretrain epoch {epoch+1}/{n_epochs}] "
                  f"recon_loss={epoch_loss/max(n_batches,1):.4f}")
    return model


def train_member(member_idx: int, train_samples, val_samples, style_weights,
                 out_dir: str, ckpt_name: str,
                 n_epochs: int, lr: float, batch_size: int, patience: int,
                 hidden_dim: int, num_layers: int, bidirectional: bool,
                 label_smoothing: float, dropout: float,
                 pretrain_epochs: int, window: int, device: str) -> dict:
    """Train one ensemble member end-to-end.

    Returns a dict with the best-checkpoint-loaded model and metadata.
    """
    from models.intent_style import IntentStylePredictor

    model = IntentStylePredictor(
        input_dim=12, hidden_dim=hidden_dim, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [member {member_idx}] params={n_params/1e6:.2f}M  "
          f"input=12 hidden={hidden_dim} layers={num_layers} "
          f"bidirectional={bidirectional} dropout={dropout}")

    if pretrain_epochs > 0:
        print(f"  [member {member_idx}] pretraining {pretrain_epochs} epochs ...")
        pretrain_lstm(model, train_samples, n_epochs=pretrain_epochs,
                      batch_size=batch_size, device=device, window=window)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5,
    )
    intent_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    style_criterion = nn.CrossEntropyLoss(weight=style_weights,
                                          label_smoothing=label_smoothing)

    log_path = os.path.join(out_dir, f"member{member_idx}_train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                "val_intent_acc", "val_style_acc"])

    ckpt_path = os.path.join(out_dir, ckpt_name)
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    last_completed_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start:start + batch_size]
            if not batch:
                continue
            max_len = max(len(s[0]) for s in batch)
            x = np.zeros((len(batch), max_len, 12), dtype=np.float32)
            intent_labels = np.zeros(len(batch), dtype=np.int64)
            style_labels = np.zeros(len(batch), dtype=np.int64)
            for i, (h, il, sl) in enumerate(batch):
                x[i, :len(h)] = h
                intent_labels[i] = il
                style_labels[i] = sl
            x_t = torch.FloatTensor(x).to(device)
            il_t = torch.LongTensor(intent_labels).to(device)
            sl_t = torch.LongTensor(style_labels).to(device)

            ip, sp, _, _ = model(x_t, None)
            ip_last = ip[:, -1, :]
            sp_last = sp[:, -1, :]

            loss = intent_criterion(ip_last, il_t) + style_criterion(sp_last, sl_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_intent, correct_style, total = 0, 0, 0
        n_vbatches = 0
        with torch.no_grad():
            for start in range(0, len(val_samples), batch_size):
                batch = val_samples[start:start + batch_size]
                if not batch:
                    continue
                max_len = max(len(s[0]) for s in batch)
                x = np.zeros((len(batch), max_len, 12), dtype=np.float32)
                intent_labels = np.zeros(len(batch), dtype=np.int64)
                style_labels = np.zeros(len(batch), dtype=np.int64)
                for i, (h, il, sl) in enumerate(batch):
                    x[i, :len(h)] = h
                    intent_labels[i] = il
                    style_labels[i] = sl
                x_t = torch.FloatTensor(x).to(device)
                il_t = torch.LongTensor(intent_labels).to(device)
                sl_t = torch.LongTensor(style_labels).to(device)
                ip, sp, _, _ = model(x_t, None)
                ip_last, sp_last = ip[:, -1, :], sp[:, -1, :]
                loss = intent_criterion(ip_last, il_t) + style_criterion(sp_last, sl_t)
                val_loss += loss.item()
                n_vbatches += 1
                correct_intent += (ip_last.argmax(dim=-1) == il_t).sum().item()
                correct_style += (sp_last.argmax(dim=-1) == sl_t).sum().item()
                total += len(batch)

        avg_val_loss = val_loss / max(n_vbatches, 1)
        intent_acc = correct_intent / max(total, 1)
        style_acc = correct_style / max(total, 1)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss,
                                    intent_acc, style_acc])

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({"model": model.state_dict()}, ckpt_path)
        else:
            epochs_since_improve += 1

        scheduler.step()
        last_completed_epoch = epoch

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    [member {member_idx}] epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                  f"intent_acc={intent_acc:.3f} style_acc={style_acc:.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience > 0 and epochs_since_improve >= patience:
            print(f"  [member {member_idx}] early stop at epoch {epoch+1} "
                  f"(no improvement in {patience}; best={best_val_loss:.4f} "
                  f"at epoch {best_epoch+1}).")
            break

    print(f"  [member {member_idx}] done: best val_loss={best_val_loss:.4f} "
          f"at epoch {best_epoch+1}, trained {last_completed_epoch+1} epochs.")

    # Reload best ckpt for downstream evaluation
    if os.path.isfile(ckpt_path):
        try:
            data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(data["model"])
        except Exception as e:
            print(f"  WARN [member {member_idx}]: could not reload best ckpt: {e}")

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "n_epochs_trained": last_completed_epoch + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train IntentStylePredictor v9 ensemble (Phase 0F)")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Per-member classification epochs (hard ceiling).")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stop after N epochs without val_loss improvement.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", default="results/intent_v9_final")
    parser.add_argument("--scenario", default="3",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4",
                                 "2_dense", "3_dense", "4_dense"])
    parser.add_argument("--scenarios", nargs="+", default=None,
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4",
                                 "2_dense", "3_dense", "4_dense"],
                        help="One or more scenarios for round-robin data collection.")
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=50,
                        help="History window cap (also pretraining window length).")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=lambda s: str(s).lower() != "false",
                        default=True)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026,
                        help="Seed for train/val split (kept fixed across ensemble members).")
    args = parser.parse_args()

    from env.sumo_env import SumoEnv

    scenarios = args.scenarios if args.scenarios else [args.scenario]
    print(f"Collecting data from {args.n_episodes} episodes "
          f"round-robin over scenarios {scenarios}...")
    envs = [SumoEnv(scenario_name=s, ego_maneuver=args.ego_maneuver,
                    use_gui=False, use_intent=False)
            for s in scenarios]
    try:
        samples = collect_intent_data(envs, n_episodes=args.n_episodes,
                                      window_cap=args.window)
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
    print(f"Collected {len(samples)} total agent trajectory samples")

    if len(samples) < 10:
        print("Not enough samples collected. Make sure SUMO is working.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Deterministic split (independent of per-member seeds)
    np.random.seed(args.split_seed)
    np.random.shuffle(samples)
    n_val = max(1, len(samples) // 5)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    style_label_counts = np.bincount(
        np.array([s[2] for s in train_samples], dtype=np.int64), minlength=3,
    )
    total_style = int(style_label_counts.sum())
    style_weights_np = np.array(
        [total_style / (3 * max(int(c), 1)) for c in style_label_counts],
        dtype=np.float32,
    )
    style_weights = torch.FloatTensor(style_weights_np).to(device)
    print(f"Style class counts: {style_label_counts.tolist()}")
    print(f"Style class weights (for CE): {style_weights_np.tolist()}")

    seeds = ENSEMBLE_SEEDS[:args.ensemble_size]
    member_meta = []
    ensemble_models = []

    for ens_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ckpt_name = f"intent_model_member{ens_idx}.pt"
        print(f"\n=== Training ensemble member {ens_idx+1}/{len(seeds)} (seed {seed}) ===")
        meta = train_member(
            member_idx=ens_idx,
            train_samples=list(train_samples),
            val_samples=val_samples,
            style_weights=style_weights,
            out_dir=args.out_dir,
            ckpt_name=ckpt_name,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            label_smoothing=args.label_smoothing,
            dropout=args.dropout,
            pretrain_epochs=args.pretrain_epochs,
            window=args.window,
            device=device,
        )

        # Per-member breakdown
        bd = evaluate_per_class(meta["model"], val_samples, args.batch_size, device)
        print(f"  [member {ens_idx}] style_overall={bd['style']['overall']:.4f} "
              f"intent_overall={bd['intent']['overall']:.4f}")
        print(f"  [member {ens_idx}] style_per_class="
              f"{[f'{a:.3f}' for a in bd['style']['per_class_acc']]} "
              f"counts={bd['style']['counts']}")
        print(f"  [member {ens_idx}] intent_per_class="
              f"{[f'{a:.3f}' for a in bd['intent']['per_class_acc']]} "
              f"counts={bd['intent']['counts']}")
        with open(os.path.join(args.out_dir, f"member{ens_idx}_breakdown.json"), "w") as f:
            json.dump(bd, f, indent=2, default=str)

        member_meta.append({
            "member_idx": ens_idx,
            "seed": seed,
            "best_val_loss": meta["best_val_loss"],
            "best_epoch": meta["best_epoch"] + 1,
            "n_epochs_trained": meta["n_epochs_trained"],
            "individual_breakdown": bd,
        })
        ensemble_models.append(meta["model"])

    # Probability-averaged ensemble eval (this is the gate-relevant breakdown)
    print("\n=== Ensemble (probability-averaged) evaluation ===")
    ens_bd = evaluate_ensemble(ensemble_models, val_samples, args.batch_size, device)
    print(f"Ensemble style overall = {ens_bd['style']['overall']:.4f}")
    print(f"Ensemble style per-class = {[f'{a:.3f}' for a in ens_bd['style']['per_class_acc']]}")
    print(f"Ensemble style counts = {ens_bd['style']['counts']}")
    print(f"Ensemble intent overall = {ens_bd['intent']['overall']:.4f}")
    print(f"Ensemble intent per-class = {[f'{a:.3f}' for a in ens_bd['intent']['per_class_acc']]}")
    print(f"Ensemble intent counts = {ens_bd['intent']['counts']}")

    with open(os.path.join(args.out_dir, "per_class_breakdown.json"), "w") as f:
        json.dump(ens_bd, f, indent=2, default=str)
    with open(os.path.join(args.out_dir, "ensemble_meta.json"), "w") as f:
        json.dump({
            "ensemble_seeds": seeds,
            "members": member_meta,
            "ensemble_breakdown": ens_bd,
            "args": vars(args),
        }, f, indent=2, default=str)

    print(f"\nEnsemble training complete. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
