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

## Lab 3 - Containerizing the model with Docker

### Question 1: What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

I registered the model of my best Lab 2 run (`secretive-stoat-955`, run ID `23b406c4191c41eb8f440320bddf3ed8`, `lr=0.0001`, `val_accuracy=0.7482`) from code with `mlflow.register_model("runs:/23b406c4191c41eb8f440320bddf3ed8/model", "food11")`. MLflow created the registered model `food11` and gave the model **version 1** (the first version of a newly created registered model). The "Models" tab shows `food11` with its latest version `Version 1`.

MLflow printed: `Run with id 23b406c4... has no artifacts at artifact path 'model', registering model based on models:/m-90b969290a40482b94e511c41514b943 instead`. This is the MLflow 3 behaviour seen in Lab 2 Q6: `log_model` stores the model as a *LoggedModel* (`m-90b96929...`) under `mlruns/1/models/`, and the registry version points to it.

- A **logged model artifact** is the output of one specific run: the files (`MLmodel`, weights, environment files) saved by `mlflow.pytorch.log_model`. It is identified by its run / model ID, every training run produces one, and it has no lifecycle of its own. It is just a record of "what this run produced".
- A **registered model** is an entry in the Model Registry with a stable, human-readable name (`food11`) that groups **versions**. Each version points to one logged model and remembers its source run, but the registry adds a management layer on top: incrementing version numbers, aliases (`champion`), tags and descriptions. Consumers reference the name/alias instead of a run ID, so the model a service uses can be changed without touching the code.

### Question 2: What aliases replaced the old built-in stages in mlflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

The fixed built-in stages `None` / `Staging` / `Production` / `Archived` are deprecated. They are replaced by **model version aliases**: free-form, user-defined names such as `champion` and `challenger` (the names used in the MLflow docs), plus free-form **tags** for any other metadata. I set the alias with `client.set_registered_model_alias("food11", "champion", 1)`, and `models:/food11@champion` now resolves to version 1.

**Why version a model separately from the run:** a run is an experiment. There are many of them (5 in Lab 2), most are thrown away, and they are identified by opaque IDs. The registry holds only the models we decided are worth deploying, under one stable name with increasing version numbers. That separates "doing experiments" from "releasing models": you get a clean release history (v1, v2, ...) with a link back to the run for lineage, and anyone can consume `food11` without knowing which of dozens of runs produced it.

**Why an alias is more flexible than a fixed stage:**
- Aliases are arbitrary names, so they can match the team's own workflow (`champion`/`challenger`, `prod-eu`, `canary`, `shadow`) instead of four hard-coded stages.
- An alias is a mutable pointer that can be moved to any version in one call. Promoting or rolling back is reassigning `champion` from v2 to v1, with no copying or re-registering, and consumers loading `models:/food11@champion` pick it up automatically.
- A version can carry several aliases at once, and an alias always points to exactly one version, so the "current production model" is never ambiguous. With stages, several versions could sit in `Production`.

### Question 3: Why load the model through an mlflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?

`src/food11/serve.py` loads the model once at startup with `mlflow.pyfunc.load_model(MODEL_URI)`, where `MODEL_URI` defaults to `models:/food11@champion` and `MLFLOW_TRACKING_URI` comes from an environment variable (default `http://127.0.0.1:5000`).

Reasons to use the registry URI rather than a file path:
- **Decoupling:** the serving code doesn't know or care where the weights are stored (local `mlruns/`, S3, a remote server...) or which run produced them. MLflow resolves `name@alias -> version -> artifact location` for us. A hard-coded `.pth` path ties the service to one file on one machine, which in particular doesn't exist inside a container.
- **Loads everything needed, not just weights:** a `.pth` file only has tensors. The service would also need the exact model class code and to rebuild the architecture. The MLflow model contains the `MLmodel` metadata, the serialized model (`data/model.pt2`), its input/output signature (`float32 [-1, 3, 128, 128] -> [-1, 11]`) and its Python requirements, and `pyfunc` gives a uniform `predict()` interface whatever the framework.
- **Governance and lineage:** what gets served is whichever version has been approved and given the `champion` alias, and each version links back to its run (params, metrics). You can always tell exactly which model is in production.

I verified the local service end to end: sending all 1096 images of `food11_processed_mini/validation` to `POST /predict` gave 820 correct answers, **accuracy 0.7482**, exactly the `val_accuracy` logged for run `23b406c4...`. So the preprocessing in `serve.py` and the class order match training.

**To serve a newer version:** register the new run's model (it becomes `food11` version 2), then move the alias: `client.set_registered_model_alias("food11", "champion", 2)`. Nothing changes in the code, the Docker image or the deployment config. The service only needs a **restart**, because it loads the model once at startup. (To pin a specific version for testing, the `MODEL_URI` environment variable could be set to e.g. `models:/food11/2` without rebuilding.)

### Question 4: Why copy `pyproject.toml`/`uv.lock` and run `uv sync` *before* copying the rest of the source code? What happens to the build cache when you only change a line in `serve.py`?

Docker caches every instruction as a layer. When it rebuilds, it reuses a layer only if the instruction **and everything before it** is unchanged (for `COPY`, the checksum of the copied files). The first layer that changes invalidates the cache for all the layers after it.

The dependencies (`pyproject.toml`/`uv.lock`) change rarely, while the source code changes all the time, so the Dockerfile goes from least to most frequently changed:

1. `COPY pyproject.toml uv.lock ./`
2. `RUN uv sync --frozen --no-dev --no-install-project`, the slow step (downloading ~1 GB of wheels, including torch)
3. (runtime stage) `COPY --from=builder /app/.venv ...`, then `COPY src/ ./src/` last

If everything were copied at once (`COPY . .` before `uv sync`), *any* edit to `serve.py` would change that `COPY` layer, invalidate `uv sync` and reinstall every dependency.

**What I measured:**
- First build: **1577 s (26 min)**, almost all of it in `uv sync` downloading packages, because the network inside Docker's WSL VM is slow (~0.4 MB/s; torch alone took ~8 minutes).
- Then I changed one line in `serve.py` (the FastAPI title) and rebuilt: **1 second**. The build log shows every step as `CACHED` (`COPY pyproject.toml uv.lock`, `uv sync`, `COPY --from=builder /app/.venv`), and only `[runtime 5/5] COPY src/ ./src/` was executed again.

I also added `RUN --mount=type=cache,target=/root/.cache/uv` to the `uv sync` step. It keeps uv's downloaded wheels in a BuildKit cache that isn't part of the image, so even when `uv.lock` changes, only new or changed packages are downloaded. I added it after a first build failed at the very end (a bytecode-compile step timed out) and all 20 minutes of downloads were thrown away with the failed layer.

### Question 5: What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.

For the naive single-stage image I built stage 1 of the same Dockerfile on its own (`docker build --target builder -t food11-api:single-stage .`). That's what a single-stage build is: the full `python:3.14` image plus uv plus the venv, all kept in the final image. (It reused the cached `uv sync` layer, so it didn't need another 26-minute download.)

| image | base | `docker images` size | real root filesystem (`du -sxh /`) | gcc inside |
|---|---|---|---|---|
| `food11-api:single-stage` | `python:3.14` | **3.51 GB** | 2.5 GB | yes |
| `food11-api:latest` (multi-stage) | `python:3.14-slim` | **2.0 GB** | 1.5 GB | no |

The multi-stage image is about **1.5 GB smaller (~43 %)**.

`docker history` shows where the space goes:
- **Biggest layer in both: the virtual environment, 1.44 GB** (`RUN uv sync ...` in the single-stage image, `COPY /app/.venv /app/.venv` in the multi-stage one). Inside it, torch alone is 711 MB, then pyarrow 152 MB, scipy 81 MB, mlflow 46 MB and pandas 42 MB.
- **Only in the single-stage image:** the build tooling of the full Python image, i.e. one `apt-get install` layer of **694 MB** (gcc, make, dev headers...), others of 202 MB and 65 MB, a larger Debian base (134 MB vs 88 MB), plus the 55 MB `uv` binary. None of this is needed to *run* the API. The multi-stage build uses these tools in stage 1 and then copies only `/app/.venv` into a slim base, so they never reach the final image.

The multi-stage image also has a smaller attack surface (no compiler or build tools) and runs as a non-root user (`app`, uid 1000).

Two choices in `pyproject.toml` also keep the image small:
- **CPU-only torch on Linux.** With the default lock, Linux `torch` pulls the CUDA 13 stack (15 `nvidia-*` packages + `triton`, ~3.6 GB of downloads), which would have made the image several GB larger with no benefit, since the container runs on CPU. I added the `pytorch-cpu` index **only for `sys_platform == 'linux'`**, so my Windows environment is unchanged (still `torch 2.14.0+cpu` from PyPI).
- `uv sync --no-dev --no-install-project` installs only the runtime dependencies.

### Question 6: What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

Before building anything, `docker build .` sends the whole **build context** (the folder) to the Docker daemon, which on Windows runs in a separate WSL VM. I measured what that would include:

| folder | files | size |
|---|---|---|
| `data/` (raw + processed Food-11) | 36,578 | 1,216 MB |
| `.dvc/` (DVC cache) | 32,042 | 1,167 MB |
| `.venv/` (Windows virtualenv) | 34,003 | 1,062 MB |
| `mlruns/` (5 logged models) | 36 | 221 MB |
| `.git/`, `mlflow.db`, ... | ~70 | ~1 MB |
| **total** | **~100,000** | **~3.7 GB** |

With the `.dockerignore`, the build log shows `transferring context: 484B` for the first build (a few kB later): only `pyproject.toml`, `uv.lock` and `src/` are sent.

- **Build speed:** without it, every build would first copy ~3.7 GB / 100k files into the VM, adding minutes even when every layer is cached. My cached rebuilds took 1-3 s. Any `COPY . .` would also change on every data or MLflow update and invalidate the cache.
- **Image size:** with my Dockerfile, which only copies specific paths (`pyproject.toml uv.lock` and `src/`), the size wouldn't change much. With a typical `COPY . .` the image would grow by up to ~3.7 GB of datasets, DVC cache, models and a useless Windows venv. It would also bake in things that don't belong in an image: the dataset, `mlflow.db`, the git history.
- **What would actually break the build: `.venv/`.** It's a *Windows* virtualenv (`Scripts/python.exe`, `.pyd`/`.dll` binaries, paths pointing to `C:\...`). If it's copied into the image, for example by a `COPY . .` after `uv sync`, it overwrites the Linux venv the builder created at `/app/.venv`. The container then has no Linux `python`/`uvicorn` under `/app/.venv/bin` and fails to start, or `uv sync` finds an incompatible environment. `__pycache__`/`*.pyc` files are compiled on the host and only add clutter. `data/`, `.dvc/` and `mlruns/` don't break anything; they make the build slow and the image huge.

A mistake I caught: I first wrote `__pycache__/` in `.dockerignore`, but that pattern only matches at the root of the context, and `src/food11/__pycache__/*.pyc` (compiled on Windows) ended up in the image. The correct patterns are `**/__pycache__/` and `**/*.py[cod]`. After fixing it, `/app/src/food11` in the image contains only `data.py`, `serve.py` and `train.py`.

### Question 7: Why can't the container simply use `127.0.0.1:5000` to reach the mlflow server on your host? What does `host.docker.internal` resolve to?

A container has its **own network namespace**, and its own loopback interface. Inside the container, `127.0.0.1` means *the container itself*, not the Windows machine. My container had the address `172.17.0.3` on Docker's bridge network, and nothing listens on port 5000 inside it. Tested from inside a container:

```
http://127.0.0.1:5000/health            -> FAILED: [Errno 111] Connection refused
http://host.docker.internal:5000/health -> OK
```

Running the image with `-e MLFLOW_TRACKING_URI=http://127.0.0.1:5000` fails at startup with `ConnectionRefusedError ... /api/2.0/mlflow/registered-models/alias?name=food11&alias=champion`. (On Linux, `--network host` avoids this by sharing the host's network namespace, so there `127.0.0.1` *is* the host. Docker Desktop on Windows/Mac runs containers inside a VM, which is why it provides `host.docker.internal` instead.)

**`host.docker.internal`** is a special DNS name that Docker Desktop provides inside containers. In my container, `getent hosts host.docker.internal` returned **`192.168.65.254`**, an address on Docker Desktop's internal VM network that Docker forwards to the Windows host. That's how the container reaches the MLflow server listening on the host's `127.0.0.1:5000`.

Making the container load the model took two more fixes that the lab's command alone doesn't cover:

1. **MLflow 3 rejected the request:** `403 'Invalid Host header - possible DNS rebinding attack detected'`. The MLflow 3 server only accepts requests whose `Host` header is on an allow-list (localhost and private IPs by default), and the container sends `Host: host.docker.internal:5000`. Fix: start the server with `--allowed-hosts`:

   ```
   uv run mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000"
   ```

2. **The metadata was found but not the model files:** `MlflowException: No such artifact`. The tracking server only returns *metadata*. For the model files, the registry tells the client to read them directly from the artifact location, which with `--default-artifact-root ./mlruns` is a host path: `file:C:/Users/Pc/Projects/mlops-lab-1/mlruns/1/models/m-90b96929.../artifacts`. That path doesn't exist in a Linux container (on Linux, MLflow turns it into the relative path `/app/C:/Users/...`). Fix: bind-mount the host's `mlruns/` at exactly that path (PowerShell):

   ```
   docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 --mount "type=bind,source=C:\Users\Pc\Projects\mlops-lab-1\mlruns,target=/app/C:/Users/Pc/Projects/mlops-lab-1/mlruns" food11-api:latest
   ```

   (The mount can't be read-only: when loading a `models:/` URI from a local store, MLflow writes a small `registered_model_meta` file next to the model. It's the same 38-byte file the local run had already written.)

With that, the container logs `Loading models:/food11@champion from http://host.docker.internal:5000` → `Model loaded`. The lab's `curl` works, and sending all 1096 validation images to the container gives **accuracy 0.7482**, the same as the local API and the training run. The cleaner long-term fix is an artifact store that the server serves over HTTP (`mlflow server --serve-artifacts --artifacts-destination ...`) or a remote store like S3/MinIO, so clients never need direct access to the server's disk.

### Question 8: Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

Yes. I stopped the container (`0a01f1904349`) and started a new one (`d329c34e2152`) from the **same image** (`6c4c207718f6`) without rebuilding. It started in ~11 s, logged `Loading models:/food11@champion from http://host.docker.internal:5000` → `Model loaded`, and predicted correctly (`Soup/9_0.jpg` → `Soup`, 0.9998).

This shows the split between image and runtime:
- **Baked into the image (immutable):** the OS and Python (`python:3.14-slim`), the dependencies (`/app/.venv`: torch, mlflow, fastapi...) and the code (`/app/src/food11/serve.py`). Inside the image, `/app` contains only `.venv` and `src/`. Searching it finds **no model weights** (no `model.pt2`, no `MLmodel`).
- **Fetched at runtime, every time a container starts:** the model. `serve.py` asks the tracking server (address from the `MLFLOW_TRACKING_URI` environment variable) which version the `champion` alias points to, then loads that version's files from the artifact store. Nothing is cached inside the container, and a stopped container's state is thrown away.

Consequences: the same image can serve a new model without rebuilding (move the `champion` alias, restart the container), and configuration comes from the environment (`-e`). But the container **depends on the tracking server and artifact store at startup**. With the MLflow server stopped or unreachable, the app fails to start (as the `127.0.0.1` test in Q7 shows). The image alone isn't enough to serve predictions.

### Question 9: The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?

The Dockerfile in git is only a *recipe*. The image I built exists only in my local Docker Desktop, as `food11-api:latest`. What's missing:

1. **A container registry.** The image has to be pushed (`docker tag` + `docker push`) to a registry that other machines can pull from: Docker Hub, GitHub Container Registry (`ghcr.io/hayamabdallatif664/food11-api`), or a cloud registry. Pushing needs `docker login` credentials, and pulling machines (CI, a Kubernetes cluster via `imagePullSecrets`) need access too.
2. **An immutable, traceable tag instead of `latest`.** `latest` is a moving label: two machines pulling it at different times can get different images. The image should be tagged with something that identifies exactly what's in it, e.g. the git commit SHA or a version (`1.0.0`), and deployments should reference the **digest** (`food11-api@sha256:...`), which never changes.
3. **An automated, reproducible build (CI).** A pipeline (e.g. GitHub Actions) that builds from a specific commit, runs tests, and pushes with the commit SHA tag. This creates the link between the git version and the image version. The build should also be reproducible: the base images (`python:3.14`, `python:3.14-slim`, `ghcr.io/astral-sh/uv:0.12.10`) could be pinned by digest (the dependencies are already pinned by `uv.lock` and `--frozen`).
4. **Model artifacts other machines can reach.** Right now the container only works on my laptop, because it needs (a) the tracking server at `host.docker.internal:5000` and (b) the host's `mlruns/` folder mounted at a Windows-derived path. A CI runner or cluster needs an MLflow tracking server on a real network address with a shared artifact store (S3/MinIO/GCS, or `--serve-artifacts`), passed in through `MLFLOW_TRACKING_URI` (plus credentials as secrets).
5. **The right platform/architecture.** The image is `linux/amd64`. ARM machines (Apple Silicon, some cloud nodes) would need a multi-arch build (`docker buildx build --platform linux/amd64,linux/arm64`).
