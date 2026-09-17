## Lab Questions and Answers

### Question 1: What files did `uv init` create and what are they used for?

Running `uv init` created the main files needed to initialize the Python project:

- `.python-version`: specifies the Python version used by the project.
- `pyproject.toml`: contains the project configuration, metadata, Python requirements, and dependencies.
- `README.md`: contains the documentation and description of the project.
- `src/mlops_lab_1/`: contains the Python source code of the project.

Later, `uv.lock` was also generated to lock the exact dependency versions and make the environment reproducible.


### Question 2: What files did `dvc init` create? Which ones should be pushed to Git?

Running `dvc init` created:

- `.dvc/config`: contains the DVC project configuration.
- `.dvc/.gitignore`: prevents DVC internal/cache files from being tracked by Git.
- `.dvcignore`: allows files and folders to be excluded from DVC operations.

The DVC configuration and metadata files should be tracked with Git, while the DVC cache and temporary files should not be pushed to GitHub.


### Question 3: Where are the DVC credentials stored when using `--global`? What other configuration levels are available?

Because the DagsHub credentials were configured using `--global`, they are stored in the user's global DVC configuration outside the Git repository.

On my Windows machine, the global DVC configuration is located under:

`C:\Users\Pc\AppData\Local\iterative\dvc`

This prevents authentication credentials from being committed to the GitHub repository.

DVC also supports project/repository configuration, local configuration using `--local`, and system-wide configuration using `--system`.


### Question 4: What happened to `.gitignore` after running `dvc add data`?

After running `dvc add data`, DVC added the following entry to `.gitignore`:

`/data`

This prevents Git from tracking and uploading the actual dataset. The dataset is managed by DVC instead, while Git only tracks the corresponding DVC metadata.


### Question 5: What information is stored inside `data.dvc`?

The `data.dvc` file contains metadata describing the tracked data directory.

In my case, it contained information such as:

- the MD5 hash identifying the dataset version,
- the total size of the tracked data,
- the number of files,
- the hash algorithm,
- and the tracked path (`data`).

The `data.dvc` file is therefore a small pointer to a specific version of the dataset and should be tracked with Git.


### Question 6: Where are the code and data stored?

The project code and DVC metadata are stored in the GitHub repository.

The actual Food-11 dataset is not stored directly in GitHub because the `data` directory is ignored by Git. Instead, the dataset files are stored in the DagsHub DVC remote.

Git therefore versions the code and `data.dvc` pointer, while DVC versions the actual dataset.


### Question 7: What happens after cloning the Git repository? How can the dataset be retrieved?

After cloning the GitHub repository into a new temporary directory, the actual `data` folder was not present. The repository contained the DVC metadata, including `data.dvc`, but not the dataset itself.

The dataset can be retrieved from the configured DVC remote using:

`dvc pull`

After running this command, DVC downloaded and restored the Food-11 dataset. I verified that `data/food11_raw` contained the three expected folders:

- `training`
- `evaluation`
- `validation`

This confirms that the data can be reproduced from the Git repository and the DVC remote.


### Question 8: Do you still see `food11_processed` and `food11_processed_mini` after checking out the previous commit?

No.

I first listed the commits that modified `data.dvc` using:

`git log --oneline -- data.dvc`

The relevant commits were:

`15ee54e Add food11_processed and food11_processed_mini`

`b14bc6d Track data folder with dvc`

I then checked out the older commit and synchronized the data:

`git checkout b14bc6d`

`dvc checkout`

After doing this, only `food11_raw` remained in the `data` directory. The `food11_processed` and `food11_processed_mini` folders were no longer present because the older `data.dvc` referenced the previous version of the data.

Finally, after running:

`git checkout main`

`dvc checkout`

the `food11_processed` and `food11_processed_mini` folders were restored.

This demonstrates that Git and DVC can be used together to switch both the code and the associated dataset between different versions.

## Lab 2 - Model training and experiment tracking with MLflow

### Question 1: Look at `pyproject.toml` and `uv.lock`. What changed?

After running `uv add mlflow torch torchvision scikit-learn`, `pyproject.toml` was updated with the new direct dependencies required for training and experiment tracking (`mlflow`, `torch`, `torchvision`, `scikit-learn`). `uv.lock` was also updated with the exact resolved package versions and all of their transitive dependencies, making the environment reproducible.

Note: on Windows the default PyPI `torch` wheel is the CPU build (`torch 2.14.0+cpu`), so training in this lab runs on the CPU. The mini dataset trains fine on CPU.

### Question 2: What is `--backend-store-uri` used for? What is `--default-artifact-root` used for? What is the difference between the metadata mlflow stores and the artifacts it stores?

`--backend-store-uri` specifies where MLflow stores experiment and run metadata. In this lab, `sqlite:///mlflow.db` stores information such as experiments, runs, parameters, metrics, timestamps, and tags in a local SQLite database.

`--default-artifact-root` specifies where MLflow stores files generated by runs. In this lab, artifacts are stored in `./mlruns`. Artifacts can include trained models, plots, images, and other output files.

Metadata describes an experiment/run and its results (small, structured, queryable values), while artifacts are the actual files produced during the run.

### Question 3: Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why shouldn't they be tracked by dvc either?

`mlflow.db` and `mlruns/` should not be tracked by Git because they are local runtime outputs generated by MLflow and change frequently as experiments are executed. They are also binary/large files (a SQLite database and saved models), which Git handles poorly.

They should not be tracked by DVC either because DVC is being used to version the project datasets, while MLflow already manages experiment metadata and artifacts (and can be pointed to a remote tracking server and artifact store when needed). Tracking MLflow outputs with DVC would unnecessarily duplicate responsibilities.

### Question 4: What happens the first time you call `set_experiment` with a name that doesn't exist yet?

Before the call, the tracking server only had the `Default` experiment (id `0`). When `mlflow.set_experiment("food11")` was called for the first time, MLflow printed:

`INFO mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.`

It automatically created a new experiment named `food11` with id `1` and artifact location `mlruns/1`, and set it as the active experiment so that subsequent `mlflow.start_run()` calls log into it. The new experiment immediately appeared in the MLflow UI next to `Default`. Calling `set_experiment("food11")` again simply reuses the existing experiment (same id) instead of creating a duplicate.

### Question 5: What is the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument and `log_param` doesn't?

`mlflow.log_param` records a *parameter*: a value that is chosen before training and stays fixed for the whole run (learning rate, batch size, number of epochs, model architecture, dataset...). A param is written once per run and cannot be changed afterwards.

`mlflow.log_metric` records a *metric*: a numeric value produced during or after training that can evolve over time (train loss, validation loss, validation accuracy...). The same metric name can be logged many times in one run.

`log_metric` takes a `step` argument because a metric is a time series: each logged value needs to be associated with a point in training (here the epoch number) so MLflow can plot the metric curve and compare values at the same step across runs. A param has a single, fixed value for the run, so there is no notion of "when" it was logged and no `step` is needed.

### Question 6: Open the run in the mlflow UI. Find the params, the metric charts, and the logged model artifact. Where does the model artifact actually live on disk?

I opened the first run (`intelligent-seal-59`, run ID `c0104108c1a5486eb3043d80fb218e30`) in the `food11` experiment:

- **Params**: on the run's *Overview* tab, the "Parameters (12)" table lists everything logged with `mlflow.log_params` at the start of the run: `dataset=mini`, `epochs=5`, `lr=0.001`, `batch_size=32`, `seed=42`, `model=resnet18`, `pretrained_weights=ResNet18_Weights.IMAGENET1K_V1`, `optimizer=Adam`, `device=cpu`, `train_images=1100`, `val_images=1096`, `test_images=1096`.
- **Metric charts**: the *Model metrics* tab shows one chart per metric. `train_loss`, `val_loss` and `val_accuracy` are line charts with the epoch on the x-axis (`step` 0-4), because they were logged once per epoch with `step=epoch`. `test_accuracy` and `test_loss` were logged once at the end without a step, so they show a single value (0.61 and 1.40).
- **Logged model artifact**: the *Artifacts* tab shows the files of the logged model: `MLmodel`, `data/model.pt2` (the serialized network weights, ~45 MB), `conda.yaml`, `python_env.yaml`, `requirements.txt`, `input_example.json` and `serving_input_example.json`. The UI notes "You're viewing artifacts assigned to a logged model associated with this run", and the Overview tab lists it under "Logged models (1)".

**Where the model lives on disk.** The tracking server was started with `--default-artifact-root ./mlruns`, so artifacts are written to the local `mlruns/` folder, not into `mlflow.db`. Because this is MLflow 3, `mlflow.pytorch.log_model` creates a *LoggedModel* entity with its own model ID, and its files are stored under the experiment's `models/` directory rather than under the run's own artifact folder:

```
mlruns/1/models/m-fb7352d44a3c43f9b6b44718e18cc032/artifacts/
├── MLmodel
├── conda.yaml
├── data/model.pt2          <- the trained weights (45 MB)
├── input_example.json
├── python_env.yaml
├── requirements.txt
└── serving_input_example.json
```

`1` is the `food11` experiment ID and `m-fb7352d44a3c43f9b6b44718e18cc032` is the logged model ID. The `MLmodel` file records `run_id: c0104108c1a5486eb3043d80fb218e30`, which links the model back to its run, and the run's own `artifact_uri` (`mlruns/1/c0104108c1a5486eb3043d80fb218e30/artifacts`) is empty. Everything else about the run (params, metrics, tags, status, the link to the model) is metadata stored in `mlflow.db`, which is why there is no per-run folder under `mlruns/1/` apart from `models/`.

### Question 7: Select these runs and click "Compare". Which learning rate gave the best `val_accuracy`? Is higher always better?

I ran the four comparison experiments sequentially (5 epochs each, mini dataset, seed 42, CPU), selected them in the `food11` runs table and clicked **Compare**. The compare page ("Comparing 4 Runs from 1 Experiment") shows the params side by side with "Show diff only" highlighting that only `lr` and `batch_size` differ, and the final metrics of each run:

| run name | run ID | lr | batch_size | val_accuracy | val_loss | train_loss | test_accuracy |
|---|---|---|---|---|---|---|---|
| secretive-stoat-955 | `23b406c4191c41eb8f440320bddf3ed8` | 0.0001 | 32 | **0.748** | 0.812 | 0.095 | 0.779 |
| traveling-wren-28 | `840ce7ca8b294deba1bf3a71a2295d7d` | 0.001 | 32 | 0.571 | 1.535 | 0.575 | 0.608 |
| merciful-mole-802 | `eebe6211bba94d3fb76dcfe62b4cdac3` | 0.001 | 64 | 0.551 | 1.852 | 0.419 | 0.581 |
| entertaining-ox-266 | `f131eea7a6844e239971b1019107bf74` | 0.01 | 32 | 0.136 | 2.421 | 2.433 | 0.136 |

The best `val_accuracy` was obtained with the **smallest learning rate, `lr = 0.0001`** (0.748, versus 0.571 for `lr = 0.001` and 0.136 for `lr = 0.01`).

No, a higher learning rate is not always better. With `lr = 0.01` the fine-tuning of the pretrained ResNet-18 diverged: the validation loss jumped to 10.0 after the first epoch and the accuracy stayed around 0.09-0.14, which is barely above chance for 11 classes (1/11 = 0.09). The large steps destroyed the useful pretrained weights. `lr = 0.001` learned but the validation loss was unstable across epochs (5.16 -> 1.77 -> 2.02 -> 1.91 -> 1.53). `lr = 0.0001` produced smooth, monotonic improvement (val_accuracy 0.64 -> 0.70 -> 0.74 -> 0.75 -> 0.75). Since the network starts from good ImageNet weights, small updates are enough to adapt it; a learning rate that is too high overshoots and makes training unstable, while one that is too low would simply learn more slowly. The right learning rate is a trade-off that has to be found experimentally, which is exactly what experiment tracking is for.

Side note: the development run `intelligent-seal-59` (`lr=0.001`, `batch_size=32`) and `traveling-wren-28` used the same hyperparameters and the same seed, and produced identical metrics, confirming that the training script is reproducible.

### Question 8: Use the parallel coordinates plot on the compare page to look at `lr`, `batch_size` and `val_accuracy` together. What pattern do you see?

On the compare page I set the parallel coordinates plot to the parameters `batch_size` and `lr` and the metric `val_accuracy`. Each run is a line going through the three vertical axes, coloured by its `val_accuracy` (blue = low, red = high; the colour scale goes from 0.136 to 0.748).

Pattern observed:

- **`lr` is the dominant factor.** The line that goes to the top of the `lr` axis (`0.01`) drops to the very bottom of the `val_accuracy` axis (0.136, dark blue). The line that goes to the bottom of the `lr` axis (`0.0001`) ends at the very top of the `val_accuracy` axis (0.748, red). The two `lr = 0.001` lines end in the middle (0.571 and 0.551, yellow). So the lines "cross": lower `lr` maps to higher `val_accuracy` in the range we tested, and the relation is monotonic.
- **`batch_size` has only a small effect.** The two runs with `lr = 0.001` differ only in batch size (32 vs 64) and their lines arrive almost at the same place on the `val_accuracy` axis (0.571 vs 0.551), i.e. within a few percent of each other. Going from 32 to 64 was slightly worse in this run, which is consistent with a larger batch meaning fewer optimizer updates per epoch (18 instead of 35 on 1100 images) at the same learning rate.
- Because three of the four runs share `batch_size = 32`, the plot cannot tell us much about batch size beyond that single comparison; the learning rate axis is where the spread is.

In short: with the pretrained ResNet-18, `val_accuracy` is driven almost entirely by the learning rate (smaller is better among 0.01 / 0.001 / 0.0001), and the batch size is a second-order effect.

### Question 9: Sort the runs table by `val_accuracy` descending. Which run is the best one? Note its run ID.

Sorting the `food11` runs table by `val_accuracy` descending gives:

| # | run name | run ID | lr | batch_size | val_accuracy | test_accuracy |
|---|---|---|---|---|---|---|
| 1 | **secretive-stoat-955** | **`23b406c4191c41eb8f440320bddf3ed8`** | 0.0001 | 32 | **0.7482** | 0.7792 |
| 2 | traveling-wren-28 | `840ce7ca8b294deba1bf3a71a2295d7d` | 0.001 | 32 | 0.5712 | 0.6077 |
| 2 | intelligent-seal-59 | `c0104108c1a5486eb3043d80fb218e30` | 0.001 | 32 | 0.5712 | 0.6077 |
| 4 | merciful-mole-802 | `eebe6211bba94d3fb76dcfe62b4cdac3` | 0.001 | 64 | 0.5511 | 0.5812 |
| 5 | entertaining-ox-266 | `f131eea7a6844e239971b1019107bf74` | 0.01 | 32 | 0.1359 | 0.1359 |

The best run is **`secretive-stoat-955`** (`--dataset mini --epochs 5 --lr 0.0001 --batch-size 32`) with `val_accuracy = 0.7482` and `test_accuracy = 0.7792`.

**Best run ID (needed in the next lab): `23b406c4191c41eb8f440320bddf3ed8`**

Its logged model lives at `mlruns/1/models/` under the model ID shown on the run's *Artifacts* tab, and can be loaded with `mlflow.pytorch.load_model("runs:/23b406c4191c41eb8f440320bddf3ed8/model")`.
