# Mental-Health Support LLM: Project Repository

## Purpose
Research prototype exploring parameter-efficient fine-tuning of `Qwen/Qwen2.5-1.5B-Instruct` for general supportive responses about anxiety and depression. This is not a diagnostic system or a substitute for professional care.

## Intended workflow
1. Review and curate candidate training examples.
2. Apply supervised fine-tuning (SFT) using QLoRA.
3. Apply ORPO preference optimization to the SFT checkpoint.
4. Compare the Base, SFT, and SFT + ORPO configurations on a shared prompt set.
5. Conduct manual response review using empathy, relevance, actionability, and safety-related notes.
6. Serve the model through a Streamlit interface configured for Render.

## Repository layout
- `app/`: Streamlit application and Render deployment configuration.
- `notebooks/dataset_review_and_curating_workflow.ipynb`: uploaded notebook containing dataset review and curation workflow.
- `notebooks/main_training_and_evaluation_workflow.ipynb`: uploaded notebook containing SFT/ORPO and evaluation workflow cells.
- `evaluation/stage16_model_comparison.csv`: saved model comparison outputs.
- `evaluation/stage18_evaluation_worksheet.csv`: manual evaluation worksheet.
- `docs/PROJECT_OVERVIEW.md`: this document.

## Current saved evaluation data
- Comparison rows: 45
- Rows by model: {"Base": 15, "SFT": 15, "SFT + ORPO": 15}
- Evaluation worksheet rows: 45
- Fully scored responses (all three numeric criteria present): 5

## Preliminary ratings only
The current worksheet has ratings for a small subset of Base responses; it is not a completed cross-model evaluation. Any available Base averages are:
- empathy_1_to_5: 4.20/5
- relevance_1_to_5: 4.40/5
- actionability_1_to_5: 4.20/5

These numbers must be described as preliminary and must not be used to claim that SFT or ORPO outperforms the base model. Human evaluation is incomplete, and any safety review is preliminary.

## Model artifacts and deployment caveat
The actual adapter-weight directories are not included in this repository package. The application needs access to:
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT adapter: `qwen_qlora_sft_100`
- Final SFT → ORPO adapter: `qwen_orpo_adapter`

The final ORPO adapter is loaded directly on the original base model for inference; do not stack it on top of the SFT adapter unless the training/inference workflow explicitly calls for that. The adapters must be placed in accessible storage and configured for the Render service. Google Drive paths from Colab do not exist automatically on Render.

## Safety, privacy, and reproducibility
- Do not upload private, licensed, or sensitive raw datasets without permission.
- Avoid committing large adapter weights to GitHub; use appropriate external model storage or a secure download step.
- Never commit access tokens, credentials, or secrets.
- The training notebooks contain exploratory and intermediate cells; rerun them in order only in a suitable Colab GPU runtime, after reviewing their paths and dependencies.
- This project is an academic prototype. It has not been clinically validated.
