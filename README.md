# Installation et configuration de l’environnement `lerobot` avec Aloha sur Google Colab

Ce document décrit les étapes nécessaires pour installer l’environnement `lerobot` de Hugging Face sur Google Colab, en particulier pour utiliser les environnements de simulation Aloha. Il inclut l’installation des dépendances Python, le clonage du dépôt `lerobot`, la configuration du mode headless pour MuJoCo, ainsi que l’installation obligatoire du package `gym-aloha`.

Pour commencer, il faut mettre à jour `pip` et installer les bibliothèques principales :

```bash
!pip install --upgrade pip
!pip install torch torchvision transformers datasets
!pip install --upgrade huggingface_hub
!pip install stable-baselines3
```

Ensuite, on clone le dépôt `lerobot` et on l’installe en mode éditable. Cela permet de refléter toute modification locale sans avoir besoin de réinstaller à chaque fois :

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
