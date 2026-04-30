# Public Release Checklist

Before making the `4Dvessel` repository public to accompany the publication, ensure the following steps are completed:

## Code & Structure
- [ ] **Verify License**: Ensure the `LICENSE` file matches the intended open-source software license (e.g., MIT, Apache 2.0).
- [ ] **Remove Hardcoded Local Paths**: Double-check `Preprocess/config.json` to ensure user-specific paths (like `C:\Users\elega\...`) are replaced with sensible defaults or generic placeholder text.
- [ ] **Review `.gitignore`**: Ensure `sam2/checkpoints/`, large datasets (`NeRF Data/`, `.mkv` files), and intermediate compilation artifacts (`__pycache__`) are properly excluded from version control.

## Documentation
- [ ] **Verify README Links**: Ensure all relative links to internal markdown files (e.g., `docs/pipeline_inventory.md`) work seamlessly in GitHub's markdown renderer.
- [ ] **Add Paper Reference/Citation**: Once published, add the DOI and BibTeX citation for "4D Vessel Reconstruction for Benchtop Thrombectomy Analysis" to the top-level README.

## Dependencies & Environments
- [ ] **Check SAM2 Checkpoint Link**: Verify that the instructions in `environments/README.md` correctly point users to the Meta SAM2 release page to download `sam2.1_hiera_large.pt`.
- [ ] **Verify 4DGaussians Fork/Commit**: Because 4DGaussians requires cloning an external repository, ensure you point users to the *exact commit hash* or *specific fork* used for this paper. Breaking changes in the upstream repo could break reproducibility over time.

## Data Availability
- [ ] **Host Example Dataset**: Provide a link (e.g., Zenodo, Google Drive, HuggingFace) to a minimum working example dataset (a short trimmed physical video + corresponding calibration video) so external researchers can test the pipeline without configuring hardware.
- [ ] **Host Ground Truth**: Consider uploading the Blender-generated validation dataset alongside the example physical data.
