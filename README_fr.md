# HEC Produits Derivs - Scripts Python

Scripts et notebooks Python pour les cours d'analyse des produits derives et finance.

## Structure du Projet

```
.
├── bintree/                    # Module d'arbre binomial
│   ├── forward_binomial_tree.py    # Module principal
│   ├── example_forward_binomial_tree.py
│   ├── quickstart_binomial_tree.py
│   ├── BINOMIAL_TREE_README.md
│   └── VISUALIZATION_GUIDE.md
├── theme_01/                   # Theme 1 : Contrats a terme et Probabilites
│   ├── forward_pricing_problems.ipynb
│   ├── forward_pricing_problems_fr.ipynb
│   ├── probability_problem_generator.ipynb
│   ├── probability_problem_generator_fr.ipynb
│   └── ...
├── theme_02/                   # Theme 2 : Evaluation d'options avec arbres binomiaux
│   ├── market-option-prices-binomial-model.ipynb
│   └── binomial_tree_practice_app.py
├── pyproject.toml              # Dependances du projet
├── uv.lock                     # Versions verrouillees des dependances
├── README.md                   # Version anglaise
└── README_fr.md                # Ce fichier (francais)
```

## Installation avec uv

`uv` est un gestionnaire de paquets Python rapide qui gere les dependances automatiquement.

### Installer uv

```bash
# Sur macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou avec Homebrew (macOS)
brew install uv

# Sur Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Ou avec pip (toutes plateformes)
pip install uv
```

## Executer des Scripts Python

Utilisez `uv run` pour executer n'importe quel script Python. Les dependances sont installees automatiquement.

```bash
# Executer un script du dossier bintree
uv run bintree/example_forward_binomial_tree.py

# Executer un script de theme_01
uv run theme_01/02-scrape-sp500-returns.py
```

## Executer des Notebooks Jupyter

### Option 1 : Demarrer Jupyter Lab (Recommande)

```bash
uv run jupyter lab
```

Cela ouvre Jupyter Lab dans votre navigateur. Naviguez vers le fichier notebook desire.

### Option 2 : Demarrer Jupyter Notebook Classique

```bash
uv run jupyter notebook
```

### Option 3 : Ouvrir un Notebook Specifique

```bash
# Ouvrir un notebook specifique directement
uv run jupyter notebook theme_02/market-option-prices-binomial-model.ipynb
```

## Executer Python de Maniere Interactive

```bash
# Demarrer une session Python interactive avec toutes les dependances
uv run python

# Ou demarrer IPython pour une meilleure experience interactive
uv run ipython
```

## Gestion des Dependances

```bash
# Synchroniser/installer toutes les dependances
uv sync

# Ajouter un nouveau paquet
uv add nom-du-paquet

# Mettre a jour tous les paquets
uv sync --upgrade
```

## Contenu des Cours

### Theme 1 : Contrats a Terme et Probabilites (`theme_01/`)

- **forward_pricing_problems_fr.ipynb** - Evaluation des contrats a terme pour actions, matieres premieres et devises
- **probability_problem_generator_fr.ipynb** - Problemes de probabilite avec esperances et variance
- Versions anglaises egalement disponibles

### Theme 2 : Evaluation d'Options (`theme_02/`)

- **market-option-prices-binomial-model.ipynb** - Telecharger des donnees d'options reelles, evaluer avec des arbres binomiaux, calculer la volatilite implicite, analyser le smile de volatilite
- **binomial_tree_practice_app.py** - Application Bokeh interactive pour pratiquer les calculs d'arbres binomiaux

Pour lancer l'application de pratique :
```bash
uv run bokeh serve --show theme_02/binomial_tree_practice_app.py
```

### Module d'Arbre Binomial (`bintree/`)

Une implementation complete d'arbres binomiaux forward pour :
- Evaluation d'options europeennes, americaines et bermudas
- Calcul du portefeuille de replication
- Visualisation de l'arbre
- Analyse des probabilites risque-neutre

Voir `bintree/BINOMIAL_TREE_README.md` pour la documentation detaillee.

## Exemple de Demarrage Rapide

```bash
# 1. Cloner ou telecharger ce depot
# 2. Naviguer vers le dossier
cd python-scripts

# 3. Executer un exemple
uv run bintree/quickstart_binomial_tree.py

# 4. Demarrer Jupyter pour explorer les notebooks
uv run jupyter lab
```

## Prerequis

- Python 3.9 ou superieur
- uv (installe comme indique ci-dessus)

Toutes les autres dependances sont gerees automatiquement par uv.
