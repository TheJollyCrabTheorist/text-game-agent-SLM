---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# Adaptateur LoRA — Agent de jeu textuel Qwen2.5

Adaptateur **LoRA** destiné à transformer un **Small Language Model (SLM)** de la famille
**Qwen2.5** en agent capable de jouer à des **jeux d'aventure textuels** (TextWorld).

## Description

- **Modèle de base :** `Qwen/Qwen2.5-0.5B-Instruct`
- **Type d'adaptateur :** LoRA (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`)
- **Tâche :** génération de texte — prédiction séquentielle de l'action à jouer
- **Méthode d'entraînement :** QLoRA (quantification 4-bit) + fine-tuning SFT sur des
  trajectoires de jeu (expert / walkthrough) TextWorld

## Contexte d'entraînement

Fine-tuning SFT avec prompts de planning ordonné.

## Utilisation

Avec la bibliothèque `peft`, l'adaptateur se charge directement sur le modèle de base :

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "models/qwen25-05b-textworld-ordered-planning/best_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()
```

## Évaluation

- **Résultat** : `evaluation/validation_ordered.json` — **en validation** : l’adaptateur atteint **11/15** réussites (**73,3 %**), avec environ **19,1 steps** et **24 %** d’actions invalides.

## Remarques

- Cet adaptateur s'appuie sur le chat template du modèle de base Qwen2.5
  (messages `system` / `user` / `assistant` avec tokens `<|im_start|>` / `<|im_end|>`).
- Les poids du modèle de base restent nécessaires : seul l'adaptateur LoRA est téléchargé.
