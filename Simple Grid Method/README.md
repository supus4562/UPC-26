# Goal grid lookup

Run from this folder after installing requirements.

```bash
python -m pip install -r requirements.txt
python train.py --dataset ../kicks_split --output grid_models
python predict.py --model grid_models/model.json
python evaluate.py --dataset ../kicks_split --model grid_models/model.json --output metrics_grid.json
```

Training loads only the saved training partition. Evaluation verifies the test fingerprint and rejects ID overlap. Prediction needs only the saved model. Dataset coordinates follow Team 358, with y forward. Command-line observations default to the problem frame, x forward. Use --frame paper for Team 358 inputs. Outputs use lateral and height coordinates in metres.

Projection is lateral=11*x/y and height=0.11+11*(z-0.11)/y. Only the internal goal cuts are used, so outer zones extend infinitely. Exact boundaries belong to the higher band. Labels are ordered low-left through high-right.

Four CSV tables contain t1, t2, ordered t1/t2, and global-control predictions. JSON also stores probabilities, covariance, 95% cell sets and ellipse cutoffs. The control point is the global training mean. Its hard classification uses the cell containing that mean; probability metrics use global cell frequencies.

Default smoothing adds five pseudo-observations from the global distribution to single-look rows and from the raw t2 distribution to paired rows. Use --smoothing 0 for raw frequencies. Empty rows always fall back to the parent. Five is a configurable uncalibrated choice. All moments and probabilities describe the same weighted empirical crossing distribution.

Ellipses use weighted empirical 95th percentiles of squared Mahalanobis distances, with a 1e-10 square-metre diagonal covariance floor. These describe predictive spread, not uncertainty in the estimated mean. Training coverage does not guarantee held-out coverage. The 95% cell set takes descending-probability cells until cumulative probability reaches .95; ties use label order.

Evaluation writes one aggregate JSON for clean observations and Gaussian noise (default standard deviation 0.005/sqrt(3) metres). Both scenarios share the same test shots. Metrics include exact/row/column accuracy, grid distance, macro recall over classes present, confusion matrices (true rows, predicted columns), summed multiclass Brier score, cell-set coverage and size, axis MSE, radial MSE/RMSE, average predicted axis/radial uncertainty, ellipse coverage/area and fallback frequency. Use --limit 200 for a pilot. Results describe airborne goals only. Full test arrays are loaded, but no per-shot JSON is written.

The package was exercised end to end on a small generated split. Full dataset benchmarking must be run locally. Keep this folder separate from earlier scripts with identical filenames.
