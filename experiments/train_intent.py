"""Train the IntentStylePredictor LSTM using behavior labels from rollouts.

Collects rollout data with ground-truth intent/style labels from the BehaviorSampler,
then trains the LSTM to predict those labels from observation history.
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


def collect_intent_data(env, n_episodes: int = 200, max_steps: int = 200):
    """Collect (observation_history, intent_label, style_label) tuples from env."""
    from state.builder import _rot2d, _wrap
    samples = []

    for ep in range(n_episodes):
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
                t_cpa = np.clip(-np.dot(dp, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
                p_cpa = dp + t_cpa * delta_v
                d_cpa = np.linalg.norm(p_cpa)
                z = [delta_xy[0], delta_xy[1], delta_v[0], delta_v[1], delta_psi, d_cz, d_cpa, 1.0, 0.1]
                agent_histories[aid].append(z)
                if len(agent_histories[aid]) > 20:
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


def train_intent_model(samples, out_dir: str, n_epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
    """Train IntentStylePredictor on collected samples."""
    from models.intent_style import IntentStylePredictor

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model = IntentStylePredictor(input_dim=9, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    intent_criterion = nn.CrossEntropyLoss()
    style_criterion = nn.CrossEntropyLoss()

    np.random.shuffle(samples)
    n_val = max(1, len(samples) // 5)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "intent_train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_intent_acc", "val_style_acc"])

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start:start + batch_size]
            max_len = max(len(s[0]) for s in batch)
            x = np.zeros((len(batch), max_len, 9), dtype=np.float32)
            intent_labels = np.zeros(len(batch), dtype=np.int64)
            style_labels = np.zeros(len(batch), dtype=np.int64)
            for i, (h, il, sl) in enumerate(batch):
                x[i, :len(h)] = h
                intent_labels[i] = il
                style_labels[i] = sl

            x_t = torch.FloatTensor(x).to(device)
            il_t = torch.LongTensor(intent_labels).to(device)
            sl_t = torch.LongTensor(style_labels).to(device)

            ip, sp, (h_i, h_s), _ = model(x_t, None)
            ip_last = ip[:, -1, :]
            sp_last = sp[:, -1, :]

            loss = intent_criterion(ip_last, il_t) + style_criterion(sp_last, sl_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_intent, correct_style, total = 0, 0, 0
        with torch.no_grad():
            for start in range(0, len(val_samples), batch_size):
                batch = val_samples[start:start + batch_size]
                max_len = max(len(s[0]) for s in batch)
                x = np.zeros((len(batch), max_len, 9), dtype=np.float32)
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

                correct_intent += (ip_last.argmax(dim=-1) == il_t).sum().item()
                correct_style += (sp_last.argmax(dim=-1) == sl_t).sum().item()
                total += len(batch)

        avg_val_loss = val_loss / max(total // batch_size, 1)
        intent_acc = correct_intent / max(total, 1)
        style_acc = correct_style / max(total, 1)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, intent_acc, style_acc])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"model": model.state_dict()}, os.path.join(out_dir, "intent_model.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={avg_train_loss:.4f} "
                  f"val_loss={avg_val_loss:.4f} intent_acc={intent_acc:.3f} style_acc={style_acc:.3f}")

    print(f"Best val loss: {best_val_loss:.4f}. Model saved to {out_dir}/intent_model.pt")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train IntentStylePredictor LSTM")
    parser.add_argument("--n_episodes", type=int, default=200, help="Episodes to collect data from")
    parser.add_argument("--n_epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--scenario", default="3", choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="Use scenario 3 or 4 for maximum actor diversity")
    args = parser.parse_args()

    from env.sumo_env import SumoEnv
    env = SumoEnv(scenario_name=args.scenario, use_gui=False)

    print(f"Collecting data from {args.n_episodes} episodes of scenario {args.scenario}...")
    samples = collect_intent_data(env, n_episodes=args.n_episodes)
    print(f"Collected {len(samples)} total agent trajectory samples")

    if len(samples) < 10:
        print("Not enough samples collected. Make sure SUMO is working.")
        env.close()
        return

    model = train_intent_model(samples, args.out_dir, n_epochs=args.n_epochs,
                                lr=args.lr, batch_size=args.batch_size)
    env.close()
    print("Intent model training complete.")


if __name__ == "__main__":
    main()
