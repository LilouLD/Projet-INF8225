import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple
from dm_control.rl.control import PhysicsError
import gym_aloha
from gymnasium.spaces import Dict as DictSpace, Box


class AlohaUnifiedEnv(gym.Env):
    """Environnement combinant AlohaInsertion et AlohaTransfer."""

    def __init__(self, task_type: str = "random", task_switch_prob: float = 0.5, action_clip: float = 0.95):
        """
        Initialise l'environnement unifié.

        Args:
            task_type: Type de tâche ("random", "insertion" ou "transfer")
            task_switch_prob: Probabilité de choisir transfer si task_type="random"
            action_clip: Valeur de clipping des actions
        """
        super().__init__()

        # Créer les environnements
        self.insertion_env = gym.make("gym_aloha/AlohaInsertion-v0")
        self.transfer_env = gym.make("gym_aloha/AlohaTransferCube-v0")

        # Type de tâche fixe ou aléatoire
        self.task_type = task_type
        self.task_switch_prob = task_switch_prob
        self.action_clip = action_clip

        self.current_env = None
        self.current_task = None

        # Combiner les espaces d'observation des deux environnements
        # On s'assure que les observations auront un format unifié avec l'indicateur de tâche
        if isinstance(self.insertion_env.observation_space, DictSpace):
            # Si l'espace d'observation est déjà un dictionnaire, nous y ajoutons un indicateur de tâche
            obs_spaces = dict(self.insertion_env.observation_space.spaces)
            obs_spaces['task_indicator'] = Box(low=0, high=1, shape=(2,), dtype=np.float32)
            self.observation_space = DictSpace(obs_spaces)
        else:
            # Si l'espace d'observation n'est pas un dictionnaire, on le convertit en dictionnaire
            self.observation_space = DictSpace({
                'observation': self.insertion_env.observation_space,
                'task_indicator': Box(low=0, high=1, shape=(2,), dtype=np.float32)
            })

        self.action_space = self.insertion_env.action_space

        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        # Réinitialiser le compteur d'erreurs
        self.consecutive_errors = 0

        # Sélectionner l'environnement selon le type de tâche
        if self.task_type == "insertion":
            self.current_env = self.insertion_env
            self.current_task = "insertion"
        elif self.task_type == "transfer":
            self.current_env = self.transfer_env
            self.current_task = "transfer"
        else:  # random
            # Si une graine est fournie, l'utiliser pour la sélection de tâche
            if 'seed' in kwargs:
                np.random.seed(kwargs['seed'])
                
            if np.random.random() < self.task_switch_prob:
                self.current_env = self.transfer_env
                self.current_task = "transfer"
            else:
                self.current_env = self.insertion_env
                self.current_task = "insertion"
        
        # Réinitialiser l'environnement sélectionné et obtenir l'observation
        obs, info = self.current_env.reset(**kwargs)
        
        # Enrichir l'observation avec l'indicateur de tâche
        enriched_obs = self._enrich_observation(obs)
        
        # Ajouter l'info de la tâche aux informations
        info["task"] = self.current_task
        
        return enriched_obs, info

    def _enrich_observation(self, obs):
        """Ajoute l'indicateur de tâche à l'observation."""
        # Créer un vecteur one-hot pour indiquer la tâche
        task_indicator = np.zeros(2, dtype=np.float32)
        if self.current_task == "insertion":
            task_indicator[0] = 1.0
        else:  # transfer
            task_indicator[1] = 1.0
        
        # Adapter l'observation selon sa structure
        if isinstance(self.observation_space, DictSpace) and isinstance(obs, dict):
            enriched_obs = dict(obs)
            enriched_obs['task_indicator'] = task_indicator
            return enriched_obs
        else:
            return {
                'observation': obs,
                'task_indicator': task_indicator
            }

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        # Limiter l'amplitude des actions pour éviter l'instabilité
        action = np.clip(action, -self.action_clip, self.action_clip)

        try:
            obs, reward, terminated, truncated, info = self.current_env.step(action)
            self.consecutive_errors = 0  # Réinitialiser le compteur en cas de succès

            if isinstance(info, dict):
                info["task"] = self.current_task
            else:
                info = {"task": self.current_task}
            
            # Enrichir l'observation avec l'indicateur de tâche
            enriched_obs = self._enrich_observation(obs)

            return enriched_obs, reward, terminated, truncated, info

        except PhysicsError as e:
            self.consecutive_errors += 1
            print(f"Erreur physique ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")

            # Si trop d'erreurs consécutives, forcer le changement d'environnement
            if self.consecutive_errors >= self.max_consecutive_errors:
                print("Trop d'erreurs consécutives, changement d'environnement.")
                self.current_env = self.insertion_env if self.current_task == "transfer" else self.transfer_env
                self.current_task = "insertion" if self.current_task == "transfer" else "transfer"

            # Réinitialiser et retourner un épisode terminé avec récompense négative
            obs, info = self.reset()
            info["physics_error"] = True
            return obs, -1.0, True, False, info

    def render(self):
        return self.current_env.render()

    def close(self):
        self.insertion_env.close()
        self.transfer_env.close()

