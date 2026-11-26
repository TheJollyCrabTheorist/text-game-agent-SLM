# TextGame Agent — Fine-tuning d'un Small Language Model pour jeux textuels

Agent observation → action pour TextWorld, avec Qwen2.5-0.5B / 1.5B et SFT LoRA/QLoRA. Le projet utilise une liste de commandes admissibles. Les corrections dites DAgger sont obtenues par alignement sur les trajectoires oracle, et ne couvrent pas tous les états hors trajectoire.

## Prérequis et installation

Python ≥ 3.10, GPU CUDA, PyTorch compatible CUDA, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `numpy`, `textworld` (et `tw-make` disponible dans le PATH). Les versions historiques exactes ne sont pas connues : conserver l'environnement déjà fonctionnel si possible. Figer les versions du nouvel environnement avec `python -m pip freeze > requirements-local.txt` ; ne pas présenter ce fichier comme celui des expériences historiques.

Ce patch ne contient pas les Safetensors. Conserver ou restaurer les adapters dans `models/` depuis les fichiers originaux. Exécuter toutes les commandes depuis la racine du projet. Voir `CORRECTIONS.md` pour la procédure d'application et les limites de validation.

## Contrôles hors GPU

```bash
python -m unittest discover -s tests -v
python src/validate_results.py
python -m compileall -q src tests
```

## Résultats historiques (non réexécutés avec le patch)

- Niveau 1, même split test de 15 jeux : base 0/15, ancien SFT 9/15 (`evaluation/old_qlora`), SFT objectif 13/15 (`evaluation/objective_qlora`), planning ordonné 10/15 (`evaluation/ordered_planning_validation`). Les noms des fichiers anciens ne prouvent pas le split ; celui-ci a été déterminé en comparant les identifiants avec le manifest. Un autre ancien run SFT donne 10/15. Le niveau 1 comporte 99 trajectoires oracle réussies sur 100 : `game_0057` (validation) a été exclu du SFT.
- Niveau 2, même benchmark de développement de 30 jeux : SFT 0.5B 23/30, SFT + corrections 27/30, SFT 1.5B 30/30 (154 actions, aucune invalide, mêmes séquences que l'expert). Le run base 1.5B annoncé est à 3/30, mais les métadonnées historiques codent 0.5B en dur : l'identité du modèle doit être confirmée par une nouvelle évaluation. Le rapport SFT 1.5B pointe vers un adapter configuré pour 1.5B.
- Tour 2 de DAgger : validation initiale 0,032365, puis 0,076903 et 0,078413 ; meilleur checkpoint = époque 0. Le résultat 5/5 sur les jeux frais ne démontre pas une amélioration due au second entraînement.
- Les résultats `progress` historiques proviennent de données contaminées par des actions futures. Les datasets ont été réparés ; les anciens poids et scores ne représentent pas les nouveaux datasets. Ne pas les utiliser comme preuve d'un entraînement corrigé.
- Les rapports historiques de récompense peuvent sommer des scores cumulés ; le patch enregistre désormais les différences de scores. Le taux de succès, le nombre de coups et les actions invalides ne sont pas recalculés par cette correction.

## Niveau 1 : nouveau pipeline SFT

Les jeux et les trajectoires fournis sont déjà présents. Pour repartir sur des trajectoires nouvellement collectées, utiliser le générateur et le collecteur d'origine, en écrivant dans des chemins neufs ou après sauvegarde explicite des données. Pour entraîner un modèle avec objectif (legacy), utiliser :

```bash
python src/prepare_sft_dataset.py --prompt-version legacy --output-dir data/processed_legacy_v2
python src/train_sft.py --model-name Qwen/Qwen2.5-0.5B-Instruct --prompt-version legacy --data-dir data/processed_legacy_v2 --output models/qwen05-legacy-v2
python src/evaluate_games.py --adapter models/qwen05-legacy-v2/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --split validation --prompt-version legacy --repetition-penalty 1.2 --output evaluation/v2/level1_validation.json
```

Pour le prompt ordered, utiliser `--prompt-version ordered` dans les trois commandes et d'autres chemins de sortie. Pour un 1.5B, changer le modèle de base à l'entraînement et à l'évaluation. Ces commandes constituent un nouveau run, pas une reconstruction certifiée des anciens checkpoints.

## Niveau 2 : collecte et entraînement correctif (nouveau run)

Exemple avec l'adapter objectif historique (restaurer ses poids au préalable). Les fichiers de sortie sont distincts des anciens. Les corrections utilisent exclusivement le split train, et la validation les trajectoires oracle de validation.

```bash
python src/evaluate_level2.py --adapter models/qwen25-05b-textworld-qlora-objective/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --prompt-version legacy --split train --manifest data/level2/splits.json --output evaluation/v2/level2_policy_train.json
python src/collect_level2_corrections.py --trajectories data/level2/trajectories --evaluation evaluation/v2/level2_policy_train.json --output data/level2/corrections_v2
python src/prepare_level2_finetuning.py --replay data/level2/replay_level1/train.jsonl --corrections data/level2/corrections_v2/train.jsonl --manifest data/level2/splits.json --trajectories data/level2/trajectories --output data/level2/finetuning_v2
python src/train_level2_corrections.py --initial-adapter models/qwen25-05b-textworld-qlora-objective/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --train-data data/level2/finetuning_v2/train.jsonl --validation-data data/level2/finetuning_v2/validation.jsonl --epochs 3 --output models/level2/qlora-corrective-v2
python src/evaluate_level2.py --adapter models/level2/qlora-corrective-v2/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --prompt-version legacy --split validation --manifest data/level2/splits.json --output evaluation/v2/level2_validation.json
```

Pour le tour 2, utiliser comme `--initial-adapter` le premier adapter correctif et les données du nouveau split train. Préparer également une validation distincte de ce split train. Ne jamais utiliser le test pour ajuster les hyperparamètres. `best_epoch=0` signifie que l'adapter initial est retenu.

Pour réentraîner la variante progress à partir du dataset déjà réparé :

```bash
python src/train_level2_corrections.py --initial-adapter models/level2/qlora-corrective-3epochs/best_adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --train-data data/level2/finetuning_progress_corrected/train.jsonl --validation-data data/level2/finetuning_progress_corrected/validation.jsonl --epochs 3 --output models/level2/qlora-progress-causal-v2
```

Évaluer cette variante avec `--prompt-version progress`. Le point de départ ci-dessus est un choix explicite pour un **nouveau** run, pas une attestation de la configuration historique. Ne pas réutiliser les scores des anciens modèles progress.

## Évaluation finale indépendante

Figer les adapters, prompts, pénalités, seeds, budget de coups et autres paramètres avant toute mesure sur ce nouveau jeu de test. Exemple : 200 jeux, nouvelles seeds, même protocole pour les modèles à comparer. La pénalité 1.2 ci-dessous correspond au défaut actuel du code, pas nécessairement aux paramètres des anciens runs. Utiliser une validation préalable si l'on doit la modifier.

```bash
python src/generate_advanced_games.py --base-seed 40000 --num-games 200 --output-dir data/level2_final_holdout --test-only
python src/collect_advanced_dataset.py --manifest data/level2_final_holdout/splits.json --output-dir data/level2_final_holdout/trajectories
python src/evaluate_level2.py --adapter models/qwen25-15b-textworld-qlora-1.5b/best_adapter --base-model Qwen/Qwen2.5-1.5B-Instruct --prompt-version legacy --repetition-penalty 1.2 --max-steps 50 --split test --manifest data/level2_final_holdout/splits.json --output evaluation/v2/holdout_15b_sft.json
python src/evaluate_level2.py --adapter none --base-model Qwen/Qwen2.5-1.5B-Instruct --prompt-version legacy --repetition-penalty 1.2 --max-steps 50 --split test --manifest data/level2_final_holdout/splits.json --output evaluation/v2/holdout_15b_base.json
```

Ajouter les 0.5B avec leur adapter et le même manifest ; ne pas sélectionner la meilleure configuration après avoir consulté ces résultats. Vérifier que l'oracle a réussi les jeux et rapporter toute exclusion.

## Recalcul des récompenses archivées

Le script suivant crée un **nouveau** rapport (ne pas écraser l'ancien) uniquement si les scores avant/après chaque action sont explicitement disponibles et continus. Il ne change pas le modèle annoncé et ne constitue pas une nouvelle inférence.

```bash
python src/recalculate_archived_rewards.py --input evaluation/final_level2_30games_q15b.json --output evaluation/v2/recomputed_15b_rewards.json
```

## Structure

`src/` scripts, `data/` jeux et trajectoires/datasets, `models/` adapters, `evaluation/` rapports, `results/` autres résultats, `report/` présentation, `tests/` tests de non-régression. Le ZIP de correction est un patch à superposer au projet existant, pas une distribution autonome.
