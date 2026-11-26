# TextGame Agent — Fine-tuning d'un Small Language Model pour jeux textuels

Agent jouant à des jeux d'aventure textuels (TextWorld / advanced TextWorld) à partir
d'observation → action, avec un Small Language Model (SLM) fine-tuné en LoRA/QLoRA.

## Objectifs du projet

1. Prendre en main un environnement RL textuel (TextWorld) et modéliser le format
   *observation / action / récompense*.
2. Construire un jeu de données de trajectoires (expert / oracle, walkthrough) exploitable
   pour du fine-tuning SFT.
3. Fine-tuner un SLM génératif (Qwen2.5-0.5B / Qwen2.5-1.5B) en LoRA/QLoRA pour prédire
   la prochaine action.
4. Évaluer rigoureusement : taux de complétion, nombre de coups, taux d'actions invalides,
   généralisation (level 2).

## Prérequis

- GPU (CUDA) — QLoRA 4-bit sur `Qwen/Qwen2.5-0.5B-Instruct` ou `Qwen/Qwen2.5-1.5B-Instruct`.
- Python ≥ 3.10, `torch`, `transformers`, `peft`, `bitsandbytes`, `textworld`.

## Pipeline — Niveau 1 (TextWorld easy / oracle)

```bash
# 1. Génération des jeux TextWorld (reproductible, seeds fixes)
python src/generate_games.py \
    --num-games 100 \
    --base-seed 42 \
    --split-seed 42 \
    --world-size 5 \
    --nb-objects 10 \
    --quest-length 5

# 2. Collecte des trajectoires oracle (walkthrough)
python src/collect_dataset.py \
    --manifest data/raw/splits.json \
    --output-dir data/raw/trajectories \
    --max-steps 100

# 3. Préparation du dataset SFT (observation / action)
python src/prepare_sft_dataset.py

# 4. Fine-tuning LoRA/QLoRA (SFT)
python src/train_sft.py \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --output models/qwen25-05b-textworld-qlora

# 5. Évaluation
python src/evaluate_games.py \
    --adapter models/qwen25-05b-textworld-qlora/best_adapter \
    --split validation \
    --prompt-version legacy \
    --output results/evaluation/finetuned_games.json
```

## Pipeline — Niveau 2 (TextWorld avancé, DAgger)

```bash
# 1. Génération des jeux level 2 "advanced" (mondes plus larges, quêtes plus longues)
python src/generate_advanced_games.py \
    --base-seed 10000 \
    --num-games 30 \
    --output-dir data/level2

# 2. Collecte des trajectoires expert level 2
python src/collect_advanced_dataset.py \
    --manifest data/level2/splits.json \
    --output-dir data/level2/trajectories

# 3. Évaluation du modèle level 1 sur les jeux level 2 (expert en référence)
python src/evaluate_level2.py \
    --adapter models/qwen25-05b-textworld-qlora-objective/best_adapter \
    --split test \
    --manifest data/level2/splits.json \
    --expert-trajectories-dir data/level2/trajectories \
    --output evaluation/level2_extended_test/baseline.json

# 4. Corrections DAgger (policy replays → expert corrige les états)
python src/collect_level2_corrections.py \
    --trajectories data/level2/trajectories \
    --evaluation evaluation/level2/zero_shot_legacy_train.json \
    --output data/level2/corrections

python src/train_level2_corrections.py \
    --train-data data/level2/corrections/train.jsonl \
    --output models/level2/qlora-corrective

# 5. 2e tour DAgger sur dataset level 2 frais (seeds 30000+)
python src/generate_advanced_games.py \
    --base-seed 30000 \
    --num-games 30 \
    --split-seed 42 \
    --output-dir data/level2_daggr2

python src/collect_advanced_dataset.py \
    --manifest data/level2_daggr2/splits.json \
    --output-dir data/level2_daggr2/trajectories

python src/evaluate_level2.py \
    --adapter models/level2/qlora-corrective-3epochs/best_adapter \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt-version legacy \
    --split train \
    --manifest data/level2_daggr2/splits.json \
    --expert-trajectories-dir data/level2_daggr2/trajectories \
    --output evaluation/final_level2_daggr2_policy_train.json

python src/collect_level2_corrections.py \
    --trajectories data/level2_daggr2/trajectories \
    --evaluation evaluation/final_level2_daggr2_policy_train.json \
    --output data/level2_daggr2/corrections

python src/train_level2_corrections.py \
    --train-data data/level2_daggr2/corrections/train.jsonl \
    --output models/level2/qlora-daggr2

python src/evaluate_level2.py \
    --adapter models/level2/qlora-daggr2/best_adapter \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt-version legacy \
    --split test \
    --manifest data/level2_daggr2/splits.json \
    --expert-trajectories-dir data/level2_daggr2/trajectories \
    --output evaluation/final_level2_daggr2_round2_test.json
```

## Résultats clés

- Niveau 1 (TextWorld, 15 jeux eval) : base zero-shot 0/15 → SFT 10/15 (66,7 %) → + objectif ordonné 14/15 (93,3 %) → + planning ordonné 11/15 (73,3 %).
- Niveau 2 (30 jeux eval, `data/level2_extended_test`) :
  - base 1.5B zero-shot : 3/30 (10,0 %) ;
  - 0.5B fine-tuné (SFT + corrections DAgger) : 27/30 (90,0 %) ;
  - **1.5B fine-tuné (SFT) : 30/30 (100,0 %)**, 5,13 steps / expert, 0 action invalide.

## Commandes utilitaires

```bash
python src/validate_results.py                 # validation des artefacts JSON
python src/compare_evaluations.py            # comparaison des runs d'évaluation
```

## Structure

- `src/` : scripts (génération, collecte, SFT, entraînement, évaluation, DAgger).
- `data/raw/` : jeux + trajectoires + splits ; `data/processed/` : datasets SFT ; `data/level2/`, `data/level2_daggr2/`, `data/level2_extended_test/` : datasets level 2.
- `models/` : adapters LoRA (best checkpoints + `training_history.json`).
- `evaluation/` : rapports d'évaluation JSON.
- `results/` : résultats consolidés / trajectoires / figures.
- `report/` : rendu écrit (à remplir).
- `configs/` : configuration (à remplir).

## Génération de slides / rapport

À compléter — le rendu attendu est une **présentation** + le **code**.
