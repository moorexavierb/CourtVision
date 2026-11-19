Put trained weights here (they are git-ignored); in Colab they live in `MyDrive/courtvision/models/`:

- `player_detector.pt` (Player, Ref, Ball, Hoop) and `court_keypoints.pt` (18 court landmarks), both from `notebooks/01_train_models.ipynb`
- `tuned_config.yaml`: detection thresholds tuned for tracking by the same notebook (config overrides; notebook 00 applies them)
