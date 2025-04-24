import os
import subprocess
import torch
import numpy as np
import gymnasium as gym
import gym_aloha
from stable_baselines3 import PPO
from huggingface_hub import snapshot_download

# 0. Détection du device
device = "cuda" if torch.cuda.is_available() else \
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
        else "cpu"
print(f"→ Device utilisé : {device}")

# 1. Modèle personnel (format SB3)
MODEL_ZIP_PATH = 'models/aloha_unified/best_model/best_model.zip'
print(f"[Local] Modèle SB3 à {MODEL_ZIP_PATH}")

# 2. Téléchargement des modèles pré-entraînés depuis HF
hf_repos = {
    'insertion_hf': 'lerobot/act_aloha_sim_insertion_human',
    'transfer_hf': 'lerobot/act_aloha_sim_transfer_cube_human'
}
hf_dirs = {}
for key, repo_id in hf_repos.items():
    print(f"[HF] Téléchargement de '{repo_id}'...")
    hf_dirs[key] = snapshot_download(repo_id, cache_dir=os.path.expanduser("~/.cache/hf"))
    print(f"[HF] Modèle {key} téléchargé dans {hf_dirs[key]}")

# 3. Chargement du modèle SB3
mon_model = PPO.load(MODEL_ZIP_PATH)
print(f"[Chargement] Modèle SB3 chargé depuis {MODEL_ZIP_PATH}")


# 4. Adaptation des observations pour SB3 (ajout task_indicator)
class ObservationAdapter:
    @staticmethod
    def adapt_for_sb3(obs):
        if isinstance(obs, dict) and 'task_indicator' not in obs:
            obs['task_indicator'] = np.zeros(2, dtype=np.float32)
        return obs


# 5. Évaluation du modèle SB3 sur les environnements gym_aloha
def evaluate_sb3_model(env_id, n_episodes=10):
    env = gym.make(env_id)
    rewards = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_r = 0.0

        while not (terminated or truncated):
            # Adapter l'observation pour inclure task_indicator si nécessaire
            adapted_obs = ObservationAdapter.adapt_for_sb3(obs)
            action, _ = mon_model.predict(adapted_obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r

        rewards.append(total_r)
        if info.get("success", False):
            successes += 1

    env.close()
    avg_reward = float(np.mean(rewards))
    success_rate = successes / n_episodes
    print(f"[SB3] Évaluation sur {env_id} → reward moyen = {avg_reward:.2f}, succès = {success_rate:.2%}")
    return avg_reward, success_rate


# 6. Évaluation des modèles LeRobot en utilisant leur script d'évaluation
def evaluate_lerobot_model(model_dir, env_id, n_episodes=5):
    task_name = env_id.split('/')[-1]
    output_dir = f"outputs/eval/act_aloha_{task_name.lower()}/eval"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "lerobot/lerobot/scripts/eval.py",
        f"--policy.path={model_dir}",
        f"--output_dir={output_dir}",
        "--env.type=aloha",
        f"--env.task={task_name}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.batch_size=1",
    ]

    print(f"[LeRobot] Exécution de l'évaluation pour {task_name} avec {model_dir}")
    print(f"[LeRobot] Commande: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Afficher la sortie
        print(result.stdout)
        if result.stderr:
            print(f"Erreurs: {result.stderr}")

        # Lecture du fichier de résultats produit par le script
        results_file = os.path.join(output_dir, "eval_results.json")
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
                avg_reward = eval_results.get("reward_mean", 0.0)
                success_rate = eval_results.get("success_rate", 0.0)
                return avg_reward, success_rate

        # Si pas de fichier de résultats, retourner des valeurs par défaut
        return 0.0, 0.0
    except Exception as e:
        print(f"[Erreur] Échec de l'évaluation LeRobot: {e}")
        return 0.0, 0.0


# 7. Exécuter les évaluations
env_ids = ['gym_aloha/AlohaInsertion-v0', 'gym_aloha/AlohaTransferCube-v0']
results = {}

# Évaluer le modèle SB3 sur les deux environnements
for env_id in env_ids:
    key = ('mon_model', env_id)
    results[key] = evaluate_sb3_model(env_id)

# Évaluer les modèles LeRobot sur les deux environnements (évaluation croisée)
for model_key, model_dir in hf_dirs.items():
    for env_id in env_ids:
        key = (model_key, env_id)
        print(f"\n[Cross-évaluation] Test du modèle {model_key} sur {env_id}")
        results[key] = evaluate_lerobot_model(model_dir, env_id)

# 8. Affichage du résumé
print("\n=== Résumé des performances ===")
print(f"| {'Modèle':12s} | {'Environnement':30s} | {'Reward moyen':12s} | {'Taux succès':12s} |")
print(f"|{'-' * 14}|{'-' * 32}|{'-' * 14}|{'-' * 14}|")
for (m, e), (avg, sr) in results.items():
    print(f"| {m:12s} | {e:30s} | {avg:12.2f} | {sr:11.2%} |")