# Installation et configuration de l’environnement `lerobot` avec Aloha sur Google Colab

Ce document décrit les étapes nécessaires pour installer l’environnement [`lerobot`](https://github.com/huggingface/lerobot) de Hugging Face sur Google Colab, en particulier pour utiliser les environnements de simulation Aloha. Il inclut l’installation des dépendances Python, le clonage du dépôt `lerobot`, la configuration du mode headless pour MuJoCo, ainsi que l’installation obligatoire du package `gym-aloha`.

Pour commencer, il faut mettre à jour `pip` et installer les bibliothèques principales :

```bash
!pip install --upgrade pip
!pip install torch torchvision transformers datasets
!pip install --upgrade huggingface_hub
!pip install stable-baselines3
```

Ensuite, on clone le dépôt [`lerobot`](https://github.com/huggingface/lerobot) et on l’installe en mode éditable. Cela permet de refléter toute modification locale sans avoir besoin de réinstaller à chaque fois :

```bash
!git clone https://github.com/huggingface/lerobot.git /content/lerobot
%cd /content/lerobot
!pip install -e .
```

Il est impératif d’installer le package `gym-aloha` pour pouvoir charger correctement les environnements utilisés dans `lerobot`, même si vous n’utilisez pas la simulation MuJoCo :

```bash
!pip install gym-aloha
```

Dans le cas où vous souhaitez utiliser l’environnement Aloha pour de la simulation sur Colab, par exemple pour des tâches de reinforcement learning, il est nécessaire de configurer un environnement headless avec MuJoCo. Cela permet d’exécuter la simulation sans interface graphique (nécessaire sur Colab).

On commence par démarrer un serveur X virtuel via `pyvirtualdisplay` :

```python
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()
print("✅ Xvfb démarré, DISPLAY =", display.display)
```

Puis on configure les variables d’environnement pour forcer l’utilisation du renderer `EGL` de MuJoCo et désactiver tout rendu graphique :

```python
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISABLE_MUJOCO_RENDERING"] = "1"
```

Il faut également installer certaines bibliothèques système et Python :

```bash
!apt-get update -qq && apt-get install -y -qq libosmesa6-dev libglfw3 libglfw3-dev libglew-dev
!pip install -q pyvirtualdisplay glfw
```

Enfin, il faut patcher `glfw` pour éviter toute tentative de création de contexte graphique :

```python
import glfw

glfw.init = lambda *a, **k: True
glfw.terminate = lambda *a, **k: None
glfw.create_window = lambda *a, **k: 0
glfw.make_context_current = lambda *a, **k: None

print("✅ Headless GL/MuJoCo ready (MUJOCO_GL=egl, DISABLE_MUJOCO_RENDERING=1)")
```

Une fois tout cela installé et configuré, vous pouvez charger les environnements Aloha, importer vos modèles ou datasets Hugging Face, et lancer des entraînements ou des simulations de politique en reinforcement learning dans un environnement Colab parfaitement fonctionnel.



## 🧠 Entraînement d'un modèle PPO sur l'environnement unifié Aloha

Le script fourni permet d'entraîner une politique unifiée PPO (Proximal Policy Optimization) sur les deux tâches `AlohaInsertion` et `AlohaTransferCube` via l'environnement `AlohaUnifiedEnv`.

Il utilise la bibliothèque [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) et prend en charge :

- la parallélisation des environnements avec `SubprocVecEnv` (ou `DummyVecEnv` sur macOS),
- le support automatique des accélérateurs `CUDA`, `MPS` (pour Mac), ou CPU selon l'environnement,
- la sauvegarde régulière des checkpoints pendant l'entraînement (`CheckpointCallback`),
- l'évaluation périodique avec enregistrement du meilleur modèle (`EvalCallback`),
- la compatibilité avec la nouvelle API Gymnasium (définition de la seed dans `reset()`).

### 🔄 Exemple d’entraînement

L’appel à la fonction principale permet d’entraîner un agent avec les hyperparamètres suivants :

```python
model, model_path = train_unified_aloha_model(
    device="auto",           # Utilisation automatique de CUDA/MPS/CPU
    batch_size=64,
    n_steps=1024,
    total_timesteps=1_000_000,
    num_envs=4               # Parallélisation sur 4 environnements si disponible
)
```

Sur macOS, le script détecte automatiquement le système et bascule vers `DummyVecEnv`, tout en définissant les variables d’environnement nécessaires (`OBJC_DISABLE_INITIALIZE_FORK_SAFETY`) pour éviter les erreurs liées au fork sur les puces Apple (M1/M2/M3).

### 📁 Organisation des sorties

- `models/aloha_unified`: dossier de sauvegarde des modèles
- `logs/aloha_unified`: logs TensorBoard
- `best_model/`: modèle avec les meilleures performances pendant l’évaluation
- `emergency_save/`: sauvegarde d’urgence en cas d’erreur critique
