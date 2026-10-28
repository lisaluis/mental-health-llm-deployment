# Mental-Health Support LLM Project

This repository combines the Streamlit/Render deployment files with the original uploaded training/review notebooks and saved evaluation CSVs.

## Start here
1. Read `docs/PROJECT_OVERVIEW.md`.
2. Review the notebooks in `notebooks/`.
3. Inspect the comparison and manual evaluation worksheet in `evaluation/`.
4. Deployment files are in `app/`.

## Important
This package does not contain the trained model adapter weights or the original raw training dataset. Render cannot access a Colab-mounted Google Drive automatically. Configure adapter storage before expecting the deployed app to run the SFT and SFT + ORPO models.

See `app/README.md` for the deployment-specific instructions.
