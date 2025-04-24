import os
import torch
import platform
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from wrapper_env import AlohaUnifiedEnv
from lerobot.configs.train import TrainPipelineConfig


def make_env(rank, seed=0):
    """
    Crée une fonction de création d'environnement pour SubprocVecEnv.
    """
    def _init():
        env = AlohaUnifiedEnv(task_switch_prob=0.5)
        # Utiliser la nouvelle API Gymnasium pour définir la graine
        # La graine sera passée à reset() au lieu d'utiliser env.seed()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_unified_aloha_model(
        total_timesteps: int = 1000000,
        save_path: str = "models/aloha_unified",
        log_dir: str = "logs/aloha_unified",
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        device: str = "auto",
        num_envs: int = 4
):
    """
    Entraîne un modèle unifié pour les tâches AlohaInsertion et AlohaTransfer sur Mac M3 Pro.

    Args:
        total_timesteps: Nombre total d'étapes d'entraînement
        save_path: Chemin où sauvegarder le modèle final
        log_dir: Dossier pour les logs TensorBoard
        batch_size: Taille du batch pour l'optimisation (réduite pour M3 Pro)
        learning_rate: Taux d'apprentissage
        n_steps: Nombre d'étapes par mise à jour
        device: Périphérique pour l'entraînement ("auto", "cpu", "mps", "cuda")
        num_envs: Nombre d'environnements parallèles
    """
    # Créer les dossiers si nécessaires
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Détecter si nous sommes sur macOS (où SubprocVecEnv pose problème)
    is_macos = platform.system() == "Darwin"

    # Sur macOS, utiliser DummyVecEnv pour éviter les problèmes de fork avec CoreFoundation
    if is_macos:
        print("macOS détecté: utilisation de DummyVecEnv au lieu de SubprocVecEnv")
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    else:
        # Sur Linux/Windows, continuer à utiliser SubprocVecEnv
        if num_envs > 1:
            env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        else:
            env = DummyVecEnv([make_env(0)])

    # Créer un environnement d'évaluation séparé
    eval_env = DummyVecEnv([make_env(42)])

    # Appliquer le même wrapper VecTransposeImage à l'environnement d'évaluation pour assurer la cohérence
    # Ce wrapper est appliqué automatiquement par Stable Baselines 3 à l'environnement d'entraînement
    eval_env = VecTransposeImage(eval_env)

    # Définir le modèle PPO avec la politique adaptée
    model = PPO(
        "MultiInputPolicy",  # Politique adaptée pour les observations dictionnaires
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourager l'exploration
        # Format net_arch corrigé selon SB3 v1.8.0+ (dictionnaire direct au lieu de liste)
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])}
    )

    # Callback pour sauvegarder des checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000 // num_envs, 1),  # Ajuster selon le nombre d'environnements
        save_path=save_path,
        name_prefix="aloha_unified_model"
    )

    # Callback d'évaluation pour suivre les performances
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=max(10000 // num_envs, 1),
        deterministic=True,
        render=False
    )

    # Entraîner le modèle avec les deux callbacks
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
        print("Entraînement terminé avec succès!")
    except Exception as e:
        print(f"Erreur pendant l'entraînement: {e}")
        # Sauvegarde d'urgence en cas d'erreur
        model.save(os.path.join(save_path, "emergency_save"))
        print("Modèle sauvegardé en urgence")
        raise e

    # Sauvegarder le modèle final
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    print(f"Modèle final sauvegardé à {final_model_path}")

    return model, final_model_path


if __name__ == "__main__":
    # Vérifier si nous sommes sur macOS et définir une variable d'environnement pour améliorer la compatibilité
    if platform.system() == "Darwin":
        print("macOS détecté: configuration des variables d'environnement pour la compatibilité fork")
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

    # Vérifier les accélérateurs disponibles
    print("Vérification des accélérateurs disponibles:")
    print("- CUDA disponible:", torch.cuda.is_available())
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("- MPS disponible:", mps_available)

    # Détection automatique du meilleur périphérique
    if torch.cuda.is_available():
        device = "cuda"
    elif mps_available:
        device = "mps"  # Pour Mac M1/M2/M3
    else:
        device = "cpu"
    print(f"Utilisation du périphérique: {device}")

    try:
        # Sur macOS, réduire le nombre d'environnements pour éviter les problèmes liés au fork
        is_macos = platform.system() == "Darwin"
        num_envs = 1 if is_macos or device == "cpu" else 4
        if is_macos and num_envs > 1:
            print(f"Sur macOS, limitation à un seul environnement pour éviter les problèmes de fork")

        # Entraîner avec gestion d'erreur
        model, model_path = train_unified_aloha_model(
            device=device,
            batch_size=64 if device != "cpu" else 32,
            n_steps=1024,
            total_timesteps=1_000_000,
            num_envs=num_envs
        )
    except Exception as e:
        print(f"Erreur globale dans l'entraînement: {e}")

