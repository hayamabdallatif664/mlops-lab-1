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

## Lab 4 - Orchestrating the app with Docker Compose

The stack has three services defined in `docker-compose.yml`:
- **`mlflow`**: tracking server and registry, built from `mlflow/Dockerfile`, with data on the named volume `mlflow-data`.
- **`inference`**: the Lab 3 image `food11-api:latest`, built from the root `Dockerfile`, with `MLFLOW_TRACKING_URI=http://mlflow:5000`.
- **`frontend`**: a Streamlit upload page, built from `frontend/Dockerfile`, with `INFERENCE_URL=http://inference:8000`.

**Changes compared to the lab handout** (each one fixes a problem I actually hit):
1. **`mlflow/Dockerfile` uses `--serve-artifacts --artifacts-destination /mlflow-data/mlruns` instead of `--default-artifact-root /mlflow-data/mlruns`.** With a plain path, the registry would tell the inference container to read the model files directly from `/mlflow-data/mlruns`, a path that only exists inside the mlflow container. This is the `No such artifact` problem from Lab 3 Q7. With the artifact proxy, new runs get `mlflow-artifacts:/...` URIs and every client downloads (and uploads) model files over HTTP through `http://mlflow:5000`. My registered model's download URI was `mlflow-artifacts:/1/models/m-86de.../artifacts`.
2. **`--allowed-hosts "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"`.** The MLflow 3 server rejects requests whose `Host` header isn't on its allow-list (`403 Invalid Host header - possible DNS rebinding attack detected`, Lab 3 Q7). Inside Compose the inference service sends `Host: mlflow:5000`. I first listed fixed ports, and a test container on port 5001 got a 403, so I switched to the wildcard form the option supports.
3. **`mlflow==3.16.0` is pinned** (the same version as the client in the inference image) instead of whatever "latest" pip picks. pip also gets a **BuildKit cache mount, `--timeout 120` and `--retries 10`** instead of `--no-cache-dir`. My internet connection was very slow (~0.1-0.3 MB/s), and the first build died after 49 minutes on a `ReadTimeoutError`, throwing away every download. With the cache mount, a retry resumes instead of starting over (the cache isn't part of the image).
4. **`docker-compose.yml`: a healthcheck on `mlflow`, and `inference` waits with `depends_on: condition: service_healthy`** (see Q6). `inference` also has `image: food11-api:latest`, so Compose reuses and tags the Lab 3 image.
5. **Aliases instead of stages.** The handout talks about moving versions to `Staging`, but stages are deprecated in MLflow 3 (Lab 3 Q2) and `serve.py` loads `models:/food11@champion`. So here the `champion` alias plays the role of "Staging", and no code had to change.

The containerized MLflow starts **empty**: it has its own database on its own volume, not the Lab 2 `mlflow.db`. So I retrained the best Lab 2 configuration into it (`uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32`, logging to the published `http://127.0.0.1:5000`). It reproduced Lab 2 exactly (val_accuracy 0.7482, test_accuracy 0.7792, run `91a58e2e...`), and I registered it as `food11` **version 1** with alias `champion`.

**End-to-end check:** from inside the frontend container I sent all 1096 validation images to `http://inference:8000/predict` and got **820/1096 = 0.7482** accuracy, the same as training. In the browser at `http://127.0.0.1:8501`, uploading `noodles_6_0.jpg` showed **"Prediction: Noodles-Pasta (83.1%)"**.

(Local workaround, not part of the committed files: Docker Desktop's internal DNS failed to resolve `files.pythonhosted.org` (`[Errno -5] No address associated with hostname`), even though the Windows host and public DNS resolved it fine. I built the two new images with `docker build --add-host files.pythonhosted.org:<Fastly IP> -t mlops-lab-1-mlflow ./mlflow` (same for `frontend`), which are the image names Compose uses, and then ran `docker compose up -d`.)

### Question 1: What happens to everything written to `/mlflow-data` if you never mount a volume there and just `docker run` this image standalone?

It's written into the **container's writable layer** and is **lost when the container is removed**.

I ran the image with no volume (`docker run -d --name q1-mlflow -p 5001:5000 mlops-lab-1-mlflow`). `docker inspect` showed `0` mounts, and `docker diff` showed the database being created inside the container's own layer:

```
A /mlflow-data
A /mlflow-data/mlflow.db
```

I created an experiment `q1-persistence-test`, so the server listed `['q1-persistence-test', 'Default']`. Then I stopped and removed the container and started a new one from **the same image**. The UI and API showed only **`['Default']`**: the experiment was gone. The new container got a fresh writable layer on top of the image, and the image itself never contains runtime data.

Stopping a container (without removing it) keeps its writable layer, but `docker rm`, `docker compose down`, or recreating the container after a config or image change throws it away. That's why the database and artifacts need a volume.

### Question 2: Why a named volume here instead of a bind mount to a folder in your repo (the way you might for local dev)? Would a bind mount work just as well?

A **named volume** (`mlflow-data:`) is storage that Docker creates and manages itself, inside Docker's own data area (on Docker Desktop, inside its WSL2 VM). A **bind mount** maps a specific host folder (e.g. `./mlflow-data`) into the container.

Why a named volume fits the mlflow service better:
- **Portable compose file:** it needs no host path, so the same `docker-compose.yml` works on any machine (Windows, Mac, Linux, a CI runner) without depending on the repo's folder layout or a path that has to exist first.
- **Docker manages its lifecycle:** it's created automatically on the first `up`, survives `docker compose down` (and container removal and rebuilds), and is only deleted on purpose with `docker compose down -v` / `docker volume rm`. It's also easy to back up and inspect with `docker volume` commands.
- **Ownership and permissions:** Docker initialises the volume with the container's view of the files (root inside the container), so there are no host-user/UID permission mismatches. With a bind mount on Linux, files created by root in the container end up root-owned in your repo.
- **Performance and correctness on Windows/Mac:** a bind mount from the Windows filesystem into the Linux VM goes through a file-sharing layer that is much slower and has weaker file-locking semantics. That's bad for SQLite (`mlflow.db`), which relies on file locks. A named volume lives natively on the VM's Linux filesystem.
- **Keeps runtime data out of the repo:** the database and model artifacts are runtime state, not source code. Just like `mlflow.db`/`mlruns/` in Lab 2, they shouldn't sit in the git working tree, where they'd need `.gitignore` and `.dockerignore` entries to stay out of commits and build contexts.

**Would a bind mount work?** Functionally, yes: `./mlflow-data:/mlflow-data` would also persist the database and artifacts across `down`/`up`. It's handy for local development when you *want* to see or edit the files directly on the host. But it ties the compose file to a host path, is slower (and riskier for SQLite) on Docker Desktop, can cause permission problems, and puts runtime data inside the repository. For a service's own state, a named volume is the better default.

### Question 3: In this lab, `MLFLOW_TRACKING_URI` will simply be `http://mlflow:5000`. Why does that hostname resolve now when it didn't before?

Docker Compose creates a **user-defined bridge network** for the project (`mlops-lab-1_default`, 172.19.0.0/16) and attaches every service to it. On user-defined networks, Docker runs an **embedded DNS server** (`127.0.0.11`, the `nameserver` in each container's `/etc/resolv.conf`). It resolves each **service name** (and container name) to that container's IP on the network.

From inside the frontend container:

```
mlflow     -> 172.19.0.2
frontend   -> 172.19.0.4
GET http://mlflow:5000/health -> OK
```

In Lab 3, the container ran on Docker's *default* bridge network, which has no service-name DNS, and the MLflow server wasn't a container at all. It was a process on the Windows host, reachable only through the special `host.docker.internal` name. A container that isn't on the Compose network still can't resolve the name: `docker run --rm python:3.11-slim ...gethostbyname('mlflow')` → `Name or service not known`.

Two more details I observed:
- The embedded DNS only answers for **running** containers. While `inference` was stopped, `inference` didn't resolve from the frontend.
- The MLflow server must also accept the new `Host` header `mlflow:5000`, hence `--allowed-hosts` (change 2 above). Without it, the name would resolve but the server would answer 403.

### Question 4: Why does the frontend read `INFERENCE_URL` from an environment variable instead of hardcoding `http://inference:8000`?

Because the name `inference` only exists **inside this Compose network**. Configuration that depends on where the container runs belongs in the environment, not in the image ("build once, configure at run time").

I ran the frontend image standalone with `docker run` and a small probe of `INFERENCE_URL + "/health"`:

| run | `INFERENCE_URL` | result |
|---|---|---|
| no env var (default) | `http://127.0.0.1:8000` | `ConnectionError ... Failed to establish a new connection` (inside the container, 127.0.0.1 is the container itself, Lab 3 Q7) |
| hardcoded Compose name | `http://inference:8000` | `NameResolutionError ... Failed to resolve 'inference'` (no Compose DNS outside the Compose network) |

With a hardcoded `http://inference:8000`, the image would only work inside a Compose project whose service happens to be called `inference`. With an environment variable, the **same image** works everywhere: in Compose (`INFERENCE_URL: http://inference:8000`), standalone against the Lab 3 container (`-e INFERENCE_URL=http://host.docker.internal:8000`), in Kubernetes (`http://food11-inference.default.svc:8000`), or locally with `streamlit run` and the default `127.0.0.1`. Nothing is rebuilt just to point it somewhere else. It's the same reasoning as `MLFLOW_TRACKING_URI` in `serve.py`.

### Question 5: Only `mlflow` and `frontend` publish a port to the host. `inference` doesn't. Why not, and how does the frontend still reach it?

`ports:` publishes a container port **on the host** (here `0.0.0.0:8501` and `0.0.0.0:5000`). That's only needed for traffic coming from *outside* Docker: a human's browser opening the frontend, the MLflow UI, and in my case `train.py` on Windows logging runs to `http://127.0.0.1:5000`.

Nobody outside the stack needs to call the inference API directly. Only the frontend does, and that traffic stays **inside the Compose network**: the frontend calls `http://inference:8000`, Compose DNS resolves `inference` to its container IP, and the request goes straight to port 8000 on that container. Containers on the same network can reach each other on any port the process listens on; publishing isn't involved.

Not publishing it is also good practice. It reduces the attack surface (the model API can't be reached from the host or LAN), avoids port clashes on the host, and forces all traffic through the intended entry point.

I tested the API from inside the network (`docker compose exec frontend python ... http://inference:8000/predict`), since it can't be reached from the host.

### Question 6: `depends_on` only waits for the mlflow *container process* to start, not for the tracking server inside it to be ready. If `serve.py` tries to load the model at startup and mlflow isn't ready yet, what happens to the `inference` container?

`serve.py` loads the model in FastAPI's startup (`lifespan`). If that raises an exception, uvicorn prints `ERROR: Application startup failed. Exiting.` and the process exits with **code 3**. The container stops (`Exited (3)`) and, because there's no `restart:` policy, **it stays down**. The frontend keeps running, but every upload fails because `inference` doesn't answer (it doesn't even resolve while stopped, Q3).

What I observed:
- **mlflow not reachable at startup:** I stopped `mlflow` and restarted only `inference`. It didn't fail instantly. It sat in `Loading models:/food11@champion from http://mlflow:5000` for over a minute, because the MLflow client retries failed HTTP requests by itself (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`, backoff factor 2, roughly 2 minutes in total). When I started mlflow again within that window, the load succeeded (`Model loaded`). If mlflow is still not ready when the retries run out, startup fails with exit code 3.
- **mlflow ready but the model isn't there:** on the very first `docker compose up` (a fresh, empty registry), inference failed immediately: `RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found` → `Application startup failed. Exiting.` → `Exited (3)`. The same happened after `down -v` (Q10). That error isn't retried.
- mlflow takes several seconds after its container starts before `/health` answers. That's the window in which the lab's plain `depends_on: [mlflow]` would start inference too early.

**How I handled it:** a `healthcheck` on `mlflow` (a Python `urllib` call to `http://localhost:5000/health` every 5 s; the slim image has no `curl`), plus `depends_on: mlflow: condition: service_healthy` on `inference`. `docker compose up` now prints `mlflow-1 Waiting` → `mlflow-1 Healthy` before `inference-1 Starting`. Other options are a retry loop in the app or `restart: on-failure`. A restart policy alone would put the container in a crash loop while the model is missing.

### Question 7: Run `docker compose ps`. Which services have a published port listed, and which don't?

```
NAME                      IMAGE                  SERVICE     STATUS                    PORTS
mlops-lab-1-frontend-1    mlops-lab-1-frontend   frontend    Up 8 minutes              0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
mlops-lab-1-inference-1   food11-api:latest      inference   Up About a minute         8000/tcp
mlops-lab-1-mlflow-1      mlops-lab-1-mlflow     mlflow      Up 39 seconds (healthy)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

- **`frontend`** and **`mlflow`** show a host mapping (`0.0.0.0:8501->8501`, `0.0.0.0:5000->5000`, plus the IPv6 `[::]` equivalents). They're published and reachable from the host at `127.0.0.1:8501` / `127.0.0.1:5000`.
- **`inference`** shows only `8000/tcp`, with no `->` mapping. That's the port declared by `EXPOSE 8000` in the Lab 3 Dockerfile: documented and reachable from other containers on the network, but not published on the host.

That matches `docker-compose.yml` exactly: only `mlflow` and `frontend` have a `ports:` section. `ps` also shows `inference` running the Lab 3 image `food11-api:latest` and `mlflow` as `(healthy)` thanks to the healthcheck.

### Question 8: Promote a new model version. Does the prediction come from the new model version, or the old one? What single command lets you pick up the new version without rebuilding any image?

I trained a second, deliberately different model into the containerized MLflow (`--lr 0.001`, val_accuracy 0.5484), registered it as `food11` **version 2**, and moved the alias: `champion -> 2` (the "Staging" role here, see change 5).

**Before restarting**, the running service still served the **old model** (v1):
- `Noodles-Pasta/6_0.jpg` → `Noodles-Pasta`, confidence **0.831**
- whole validation set: **820/1096 = 0.7482**, v1's accuracy

`serve.py` resolves `models:/food11@champion` **once, at startup**, and keeps that model object in memory. Moving the alias in the registry doesn't notify the running process.

**The single command:**

```bash
docker compose restart inference
```

The logs show `Loading models:/food11@champion from http://mlflow:5000` → `Model loaded`, and now:
- `Noodles-Pasta/6_0.jpg` → `Noodles-Pasta`, confidence **0.9997**
- whole validation set: **601/1096 = 0.5484**, exactly v2's logged `val_accuracy`

Afterwards I moved `champion` back to v1 (the better model) and restarted again.

### Question 9: Why does `restart` alone work here, with no rebuild? What does that tell you about what's baked into the inference image versus fetched at container startup?

Because **the model isn't in the image**. As in Lab 3 Q8, the image only contains the OS, Python, the dependencies (`/app/.venv`) and the code (`serve.py`). The model is **fetched at container startup**:
1. `serve.py` reads `MLFLOW_TRACKING_URI` (`http://mlflow:5000`) and `MODEL_URI` (`models:/food11@champion`) from the environment.
2. It asks the registry which version the alias points to *right now*.
3. It downloads that version's files through the artifact proxy (`mlflow-artifacts:/...`) and loads them.

`docker compose restart` stops and starts the **same container** (same image, same config, no rebuild, no recreate), so step 2 simply gets a different answer after the alias moved. The image is a generic "food11 server" and **the registry decides which model it serves**. Releasing or rolling back a model is a registry operation plus a restart, not a code change or image build. The trade-off is that `inference` can't start without a reachable MLflow and a valid `champion` version (Q6, Q10).

### Question 10: Is your registered model and its Staging assignment still there after this `down`/`up` cycle? Now try `docker compose down -v` followed by `docker compose up`: what's different this time, and why?

**`docker compose down` → `up`: everything is still there.**
- `down` removed the three containers and the network `mlops-lab-1_default`, but the volume `mlops-lab-1_mlflow-data` still existed (`docker volume ls`).
- After `up`, the new mlflow container mounted the same volume: `food11` versions `[1, 2]` with `{'champion': '1'}`, both runs in the `food11` experiment, and inference loaded the champion again (**820/1096 = 0.7482**).

**`docker compose down -v` → `up`: everything is gone.**
- `-v` also removes the named volumes declared in the compose file: `Volume mlops-lab-1_mlflow-data Removed`.
- The next `up` printed `Volume mlops-lab-1_mlflow-data Creating`, a brand-new **empty** volume. MLflow created a fresh database: experiments `['Default']` only, registry `RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found`.
- So `inference` failed at startup again: `Application startup failed. Exiting.` → `Exited (3)`, while `mlflow` and `frontend` were up.

The containers are disposable, and the **volume is the only place the state lives**. `down` keeps it, `down -v` deletes it. In DVC terms, `down -v` is like deleting the cache and the remote: nothing is left to check out. To recover, I'd have to retrain/re-log the model and register it again (or restore a backup of the volume). The Lab 2 data on the host (`mlflow.db`, 888,832 bytes, and the 36 files in `mlruns/`) was unchanged, because it's completely separate storage.

### Question 11: This compose file is still meant to run on one machine. What would have to change for the `inference` service to run as three replicas behind a load balancer, or for the mlflow service to survive a machine failure?

Docker Compose is a **single-host** tool: every container runs on the one Docker engine of one machine, and it only restarts containers when they exit. What it can't provide:

**Three `inference` replicas behind a load balancer**
- Compose can start copies on the same machine (`docker compose up --scale inference=3`, or `deploy.replicas: 3`). Its DNS then returns all three IPs for `inference`, but that's simple DNS round-robin with no health-aware load balancing. If the machine dies, all three replicas die with it.
- What's needed is an **orchestrator** that schedules replicas across **several machines** (e.g. Kubernetes `Deployment` with `replicas: 3`), a **load balancer / Service** that spreads traffic only to *healthy* replicas (readiness probes, e.g. on `/health` once the model has loaded), and rolling updates so a new model or image can be rolled out without downtime.
- The image must come from a **registry** every node can pull from (Lab 3 Q9), not from a local build.

**`mlflow` surviving a machine failure**
- Its state currently lives in a **named volume on one machine's disk** and in a **SQLite file**. If that machine fails, the registry, runs and models are gone or unreachable. SQLite also can't be shared safely by several server instances.
- What's needed: move the state off the machine, using a **managed/replicated database** (PostgreSQL/MySQL, e.g. a cloud database with backups and failover) as `--backend-store-uri`, and an **object store** (S3, GCS, Azure Blob, MinIO) as the `--artifacts-destination`. The mlflow server itself then becomes stateless and can run as several replicas on different nodes behind a load balancer, rescheduled automatically by the orchestrator if a node dies.
- Plus the things a production setup expects and Compose doesn't give: cross-node service discovery and networking, secrets management (DB passwords, S3 keys), autoscaling, health-based self-healing across machines, and monitoring.

In short, Compose describes *what* runs together on one host. Running replicas and surviving failures needs a multi-node orchestrator (Kubernetes, or Docker Swarm with `docker stack deploy`) plus external, replicated storage for the stateful parts.

## Lab 5 - CI/CD with GitHub Actions

What I added:
- `src/food11/data.py` now exposes `CLASSES`, `category_from_filename()` and `resize_image()`, and the processing loop uses them. I checked that `resize_image` produces pixel-identical output to the old code, so the DVC-tracked data stays valid.
- `src/food11/serve.py`: when `SKIP_MODEL` is set, startup skips loading the model, `/health` still returns 200, and `/predict` returns **503** while `model is None`. I kept FastAPI's `lifespan` handler from Lab 3 (the lab's `@app.on_event("startup")` is the older, deprecated form of the same thing). `serve.py` now imports `CLASSES` from `data.py` instead of keeping a second copy.
- `tests/`: the lab's 3 data tests, `test_health`, `test_predict_returns_valid_json`, plus 2 extra API tests (503 without a model, 400 for a non-image upload), and the fixture `tests/fixtures/bread.jpg` (4 kB, in git). My `FakeModel.predict` returns **logits as a numpy array** (`[[4.0] + [0.0]*10]`), because that's what the real resnet18 pyfunc returns and `serve.py` applies the softmax itself. It also asserts the input shape is `(1, 3, 128, 128)`.
- `pyproject.toml`: dev group (`pytest`, `ruff`, `httpx`), `[tool.pytest.ini_options] pythonpath = ["."]` so tests can `import src.food11...`, and `[tool.ruff.lint.isort] known-third-party = ["mlflow"]`. Without the last one, ruff mistakes the Lab 4 `mlflow/` folder for a local module. Fixing the ruff findings also changed `train.py` (an unused variable) and `serve.py` (`Annotated[UploadFile, File()]`).
- `src/food11/validate_data.py` (data contract), `.github/workflows/ci.yml` (`test` → `smoke` → `push`), `src/food11/promote.py` (quality gate), `lab/lab5.md` (pipeline diagram).

Locally: `uv run ruff check src tests` → `All checks passed!`, `uv run pytest -q` → `7 passed` in ~5 s (no GPU, no data, no MLflow).

### Question 1: Why is it a bad idea for a unit test to load the real model from your MLflow server? List at least two ways the test could fail even though your code is correct.

A unit test should check *my code*, quickly and the same way every time, anywhere. Loading the real model makes the result depend on things outside the code under test.

Ways it fails with correct code:
1. **The MLflow server isn't reachable.** On a GitHub runner, `http://127.0.0.1:5000` is the runner itself, and my server lives on my laptop. Locally it fails whenever the server isn't running. Lab 4 Q6 showed the client then retries for about 2 minutes before failing.
2. **The registry state changed.** Someone moves `champion` to another version (Lab 4 Q8), or the alias or model doesn't exist (`RESOURCE_DOES_NOT_EXIST` after `docker compose down -v`, Lab 4 Q10). The expected prediction changes or the load fails, though no code changed.
3. **The artifacts can't be fetched from where the test runs.** In Lab 3 the registry pointed at `file:C:/Users/Pc/...`, a Windows path that doesn't exist on a Linux runner (`No such artifact`).
4. **Environment mismatch:** a different torch/mlflow version between the logged model and the test environment, or network timeouts while downloading the 45 MB model.

It's also slow (import torch, download, deserialize), so tests stop being "a second". And the result mixes two questions: "is my code right?" and "is the current model good?". The second one is the job of the evaluation and the quality gate, not of unit tests.

### Question 2: The fake model always answers "Bread" with 0.90 confidence. What exactly does `test_predict_returns_valid_json` prove, and what does it *not* prove?

**What it proves (the code around the model):**
- the app starts with `SKIP_MODEL=1`, and `/predict` is wired with the right method and multipart field name (`file`)
- an uploaded JPEG is read and decoded with PIL, and preprocessing produces exactly the tensor the model expects (my fake asserts `x.shape == (1, 3, 128, 128)`)
- the post-processing is correct: softmax over the logits, `argmax`, index 0 → `"Bread"` through `CLASSES`
- the response has the agreed JSON contract: status 200, a `category` string, a `confidence` between 0 and 1

**What it does not prove:**
- **anything about the real model**: its accuracy, or whether it would recognise this image as bread. The fake answers "Bread" for any image, so the fixture being a bread photo doesn't matter.
- that the real MLflow model **loads** and returns the same kind of output (logits as an array of shape `(1, 11)`). If the real model returned probabilities, a DataFrame or a different shape, this test would still pass.
- that preprocessing matches training (resize, normalization constants). The fake ignores pixel values.
- that the **class order** is right for every class. Only index 0 → Bread is exercised. If two other classes were swapped, the test wouldn't notice.
- integration with MLflow, the network, performance, concurrency.

So it's a **contract test of the API code**. Model quality is checked separately (evaluation, the quality gate). The integration is checked by the smoke test (for startup) and by the Lab 3/4 end-to-end runs, which gave 0.7482 accuracy over 1096 images.

### Question 3: Which check caught it? Why is a train/test duplicate more dangerous than a corrupted image?

`uv run python src/food11/validate_data.py data/food11_processed_mini` checks 3,292 images.

**The real data already fails check 4**, before any change: the original Food-11 download contains byte-identical images in different splits. On the raw files too: raw `evaluation/0_131.jpg` and raw `training/0_106.jpg` have the same md5 (`730837e9f63d...`). There are **43 cross-split duplicate groups in the mini dataset** and **611 in the full dataset** (16,643 images, checked in 2 min 44 s). The other 4 checks pass on both:

```
  [PASS] 1. each split has exactly the 11 class folders
  [PASS] 2. all 3292 images open, are RGB and 128x128
  [PASS] 3. every class has >= 50 images per split (smallest: 96)
  [FAIL] 4. no image (by content md5) appears in two splits
           - evaluation/Bread/0_131.jpg == training/Bread/0_106.jpg
           ...
  [PASS] 5. class shares train vs validation within tau=0.05 (max gap 0.003)
4/5 checks passed     (exit code 1)
```

Then I copied `training/Bread/0_0.jpg` into `evaluation/Bread/0_0_copy.jpg`. **Check 4 (no image in two splits, by md5) caught it.** The number of duplicate groups went from 43 to 44, and the new group is exactly `['evaluation/Bread/0_0_copy.jpg', 'training/Bread/0_0.jpg']`. Checks 1, 2, 3 and 5 still passed, since one extra image doesn't change folders, sizes, counts or balance in a visible way. I then deleted the copy (`evaluation/Bread` is back to 100 files).

**Why a duplicate is more dangerous than a corrupted image:**
- **A corrupted image fails loudly.** `train.py` crashes (check 2 also catches it), someone notices immediately, and it gets fixed. No wrong result is ever produced.
- **A train/test duplicate fails silently.** Training runs normally and the metrics look *good*. The model has already seen those "test" images, so test accuracy is **inflated**: it measures memorisation on part of the test set instead of generalisation. Every decision built on that number is then biased: which hyperparameters win, and which version the quality gate promotes (the gate compares `test_accuracy`). The real drop shows up only in production, on images nobody has seen.

This means my Lab 2 test accuracy (0.7792) is likely somewhat optimistic, because the evaluation split shares images with training. Fixing it means de-duplicating the raw data, re-processing, and creating a new DVC version. That's outside this lab, so I left the DVC data unchanged.

### Question 4: Your CI runner cannot see your data. Where in your workflow should `validate_data.py` run so that it still protects you? Give two places.

1. **In the data pipeline, before the data gets versioned:** right after `src/food11/data.py` produces `data/food11_processed*`, and before `dvc add data` / `git commit data.dvc` / `dvc push`. For example as a stage of a DVC pipeline (`dvc.yaml`: `process → validate → train`, so `dvc repro` stops when validation exits 1), or as a git pre-commit hook that runs when `data.dvc` changes. Bad data then never becomes a version anyone can check out.
2. **At the start of every training run, on the machine that has the data:** as the first step before `train.py` (or called inside it), right after `dvc pull` / `dvc checkout`. It also protects against a stale or partial checkout. The run stops before wasting hours of training, and the result can be logged to MLflow (e.g. a tag `data_contract=pass` on the run).

(A third option is a CI job that *does* get the data: store DagsHub credentials as GitHub secrets, `dvc pull` the mini dataset from the DagsHub remote in the runner, and run the contract on every change to `data.dvc`. Or use a self-hosted runner on the laptop. The script itself, with its PASS/FAIL logic, can also be unit-tested in CI on a tiny fake dataset built in a temp folder.)

### Question 5: The `smoke` job has `needs: test`. What happens to it if a unit test fails? What would happen without `needs`?

With `needs: test`, the jobs form a chain (`test` → `smoke` → `push`). If any step in `test` fails, the job fails and every job that depends on it is **skipped**: `smoke` never builds or starts the container, `push` doesn't publish anything, and the run is marked failed. In my successful runs they ran strictly one after another: `test` (24 s), then `smoke` (98 s), then `push` (264 s).

Without `needs`, jobs run **in parallel** on separate runners. `smoke` would build and probe the image even though the tests failed, and could report green on its own. If `push` didn't depend on anything either, it could **publish an image built from code whose tests fail**, which is exactly what CI should prevent. It also wastes runner minutes on a build that will be thrown away. `needs` turns separate checks into a gate: each stage only runs on code that passed the earlier ones.

### Question 6: How long did the whole run take? Run it a second time and compare. Which step got faster, and why?

| | run 1 (commit `7dce957`) | run 2 (empty commit `63e5c16`) |
|---|---|---|
| **total (first job start → last job end)** | **390 s (6.5 min)** | **160 s (2.7 min)** |
| `test` job | 24 s (`uv sync` 5 s, `pytest` 11 s) | 22 s (`uv sync` 3 s, `pytest` 12 s) |
| `smoke` job | 98 s (`docker build` 78 s) | 106 s (`docker build` 84 s) |
| `push` job | 264 s (`build-push-action` **230 s**) | 27 s (`build-push-action` **7 s**) |

- **The `push` job got much faster (230 s → 7 s for the build).** `docker/build-push-action` uses `cache-from: type=gha` / `cache-to: type=gha,mode=max`. Run 1 stored every image layer in the GitHub Actions cache, and in run 2 nothing had changed (empty commit), so every layer, including the 1.4 GB venv, came from the cache and only had to be pushed to GHCR, where it already existed.
- **`uv sync` got slightly faster (5 s → 3 s)** because `astral-sh/setup-uv` with `enable-cache: true` restores uv's download cache. It was already fast because Lab 3's CPU-only torch index makes the runner download the ~190 MB CPU wheel instead of ~3.6 GB of CUDA packages.
- **`smoke` did not get faster** (78 s → 84 s, just noise). It uses a plain `docker build` with no cache configured, and every run starts from an empty machine (nothing from a previous run is there unless cached). It could use `docker/setup-buildx-action` with `cache-from: type=gha` too.

### Question 7: Why tag with the commit SHA rather than `latest`? Describe a concrete situation where `latest` leads two machines to run different code.

After run 1, the `push` job published `ghcr.io/hayamabdallatif664/mlops-lab-1:7dce957b0043a1b999b4d21df0208f7a5e017691` (the package is public, since the repo is public). I pulled it on my laptop with `docker pull` and ran it the way the smoke job does: `/health` → `{"status":"ok"}`, `/predict` → `503 {"detail":"Model not loaded"}`. Its digest is `sha256:d430255af05b...`.

**Why the SHA:** a commit SHA tag is **immutable and traceable**. It always refers to the image built from exactly that commit. From any running container I can find the code (`git show 7dce957`), the CI run that built and tested it, and the tests that passed. Rolling back means deploying the previous SHA. `latest` is a **moving label**: it points to whatever was pushed last, so it says nothing about which code is inside, and its meaning changes over time.

**Concrete situation:** the inference service runs on two servers behind a load balancer, both configured with `image: ghcr.io/.../mlops-lab-1:latest`.
- Monday: server A starts and pulls `latest` = commit X.
- Tuesday: a commit Y is merged that changes preprocessing (for example new normalization constants), and CI moves `latest` to Y.
- Wednesday: server B is added (autoscaling, or a replacement after a crash) and pulls `latest` = Y. Server A still runs its cached X, since Docker doesn't re-pull an image it already has.

Both servers report the "same version" (`latest`), yet the same photo gets different predictions depending on which server answers. Reproducing a bug is guesswork, and "roll back to the previous version" is impossible, because nobody knows which code was behind `latest` on Monday. With SHA tags each server runs a version you can read off, and they only change when someone deploys a new SHA on purpose.

(This completes Lab 3 Q9: the image is now in a **registry** (GHCR), with an **immutable tag** (the commit SHA), produced by a **reproducible CI build**.)

### Question 8: Why does the `push` job declare `permissions:` explicitly? What could a compromised step do with a token that had write access to everything in the repo?

**Least privilege.** `GITHUB_TOKEN` gets the repository's *default* permissions unless the workflow says otherwise. Depending on repo and org settings, that's either read-only, which is too little for GHCR (pushing an image needs `packages: write`), or read and write to almost everything, which is far too much. A `permissions:` block sets exactly what the job gets: `contents: read` (to check out the code) and `packages: write` (to push the image), and **everything not listed becomes `none`**. The other jobs (`test`, `smoke`) never need to write anything.

**What a compromised step could do with write access to everything:** the `push` job runs third-party code, namely the actions `docker/login-action` and `docker/build-push-action` (a hijacked version tag would run attacker code), plus every package installed during `docker build`. With a write-all token, that code could:
- **push commits to the repository**, for example a backdoor in `serve.py` or `train.py`, or a modified `.github/workflows/ci.yml` that sends repository secrets to the attacker on the next run
- create or move **tags and releases**, delete branches, rewrite what people download
- open, approve, merge or comment on **pull requests and issues**, change GitHub Pages or deployments
- publish malicious **packages/images** under the project's name

With `contents: read` + `packages: write`, the worst it can do is push a bad image to this repo's package. That's still bad, but the code, history, workflows and PRs can't be changed. The token is also short-lived (it expires when the job ends), which limits how long a stolen token is useful. Pinning actions to a full commit SHA instead of `@v6` reduces this risk further.

### Question 9: What does the pull request page show? Could you still merge? What does that change for a team compared to "everybody runs the tests on their laptop before pushing"?

I protected `main` with a branch protection rule: **Require a pull request before merging**, **Require status checks to pass** with `test` and `smoke` as required checks, and **Do not allow bypassing the above settings**. GitHub's API then reports `"protected": true`, `"contexts": ["test", "smoke"]`, `"enforcement_level": "everyone"`. On the branch `break-category-test`, I made `category_from_filename` return the next class (`CLASSES[(class_index + 1) % len(CLASSES)]`), pushed, and opened **pull request #1**.

**What the PR page showed:** a red box **"Some checks were not successful — 2 failing, 4 skipped checks"**:
- `ci / test (pull_request)`: **Failing after 22s**, **Required**
- `ci / test (push)`: **Failing after 21s**, **Required**
- `ci / smoke (pull_request)` and `ci / smoke (push)`: **Skipped**, **Required** (they `needs: test`)
- `ci / push (pull_request)` and `ci / push (push)`: Skipped (not required; it only runs on `main` anyway)

There are two runs because the workflow triggers on both `push` (to the branch) and `pull_request`. The failing step was `uv run pytest -q`: `test_category_from_filename` failed because `"0_12.jpg"` became `"Dairy product"` instead of `"Bread"`.

**Could I still merge? No.** The **Merge pull request** button was greyed out. Before the rule existed, on the same PR with the same red check, the button was active and the page said "Merging can be performed automatically", so the broken code could have gone straight onto `main`. And because the rule doesn't allow bypassing, even I as repository owner couldn't force it.

I then pushed a fix to the same branch: `test` and `smoke` went green on the PR, the button became active, and the PR was merged.

**What changes for a team:** "run the tests before pushing" relies on everybody remembering, having the same environment, and being honest about it. With required checks, it's **enforced by the server, not by goodwill**:
- **Nobody can forget or skip it.** Every change to `main` has passed `test` and `smoke`, without exception, so `main` is always in a working, deployable state, and every image the `push` job builds comes from tested code.
- **Same clean environment for everyone.** Tests run on a fresh runner from `uv.lock` (`uv sync --frozen`), not on a laptop with extra packages, local data or a running MLflow server. "Works on my machine" goes away.
- **The result is visible.** Reviewers see the red or green checks on the PR before reading the code, and a failure is linked to the exact commit and log.
- **It scales.** The more people push, the more valuable an automatic gate becomes. The cost is a few minutes of waiting per PR and keeping the tests reliable, since a flaky test now blocks everyone.

One subtlety I noticed: GitHub treats a **skipped** required check as satisfied. If only `smoke` were required, a failing `test` would *skip* `smoke` and the PR would become mergeable. That's why `test` itself must also be a required check.

### Question 10: For each candidate, what did the gate decide, and with which numbers? With your test-set size, how many accuracy points does `2 * se` represent?

Current champion: `food11` **v1** (Lab 2 run `23b406c4...`, lr 1e-4), **test_accuracy = 0.7792** on **n = 1096** test images (the `test_images` param logged by `train.py`).

```
se        = sqrt(p (1 - p) / n) = sqrt(0.7792 * 0.2208 / 1096) = 0.0125
threshold = acc_champion + 2 * se = 0.7792 + 0.0251 = 0.8043
```

I trained two new runs into the local MLflow server, registered them, and ran `uv run python src/food11/promote.py --version <n>` on each:

| candidate | run | settings | test_accuracy | vs threshold 0.8043 | decision |
|---|---|---|---|---|---|
| **v2** | `f0d4ad38...` | lr **1e-5** (very small), 5 epochs | **0.5319** | 0.272 below | **not promoted** → alias `challenger`, `champion` stays v1 |
| **v3** | `f33560b4...` | lr **1e-4** (best settings), 5 epochs | **0.7792** | 0.025 below | **not promoted** → alias `challenger`, `champion` stays v1 |

In both cases `promote.py` wrote the decision and the numbers as tags on the candidate version. For v3: `gate_decision=challenger`, `gate_candidate_accuracy=0.7792`, `gate_champion_accuracy=0.7792`, `gate_champion_version=1`, `gate_se=0.0125`, `gate_threshold=0.8043`, `gate_n=1096`. Final registry state: `{'champion': '1', 'challenger': '3'}`. An alias points to a single version, so promoting v3 to challenger moved the alias away from v2, while the tags on v2 still record its decision.

- v2 is clearly worse: with lr 1e-5, five epochs aren't enough to adapt the pretrained network.
- v3 is **exactly as good** as the champion. It uses the same settings and the same `seed=42`, so training is deterministic on CPU and reproduces 0.7792. The gate refuses it, which is the point: a candidate has to be clearly better, not just equal or better by noise.

**`2 * se` = 0.0251, i.e. about 2.5 accuracy points.** With 1096 test images, a candidate needs at least **80.43%** to replace a 77.92% champion, roughly **28 more correctly classified test images**. Differences smaller than that are within the statistical noise of a 1096-image test set (a ~95% band). The leakage found in Q3 means these test accuracies are optimistic for every model.

### Question 11: `promote.py` needs to reach your MLflow server, so it cannot run in GitHub's cloud runners while MLflow lives on your laptop. Name two ways to make it run automatically after training.

1. **Move the tracking server somewhere CI can reach: DagsHub's hosted MLflow.** Every DagsHub repository comes with an MLflow server (`https://dagshub.com/hayamabdallatif664/mlops-lab-1.mlflow`), the same repo that's already my DVC remote. `train.py` logs and registers there (`MLFLOW_TRACKING_URI` + DagsHub username/token as environment variables). A GitHub Actions workflow then runs `promote.py` in a normal cloud runner, with the URI and token stored as **repository secrets**. It can be started after training by `repository_dispatch` (training sends an event when it finishes), by `workflow_dispatch` with the version number as input, or on a schedule that checks for new versions. (The Lab 4 `mlflow` container deployed on a reachable server with a proper database and object store would work the same way.)
2. **Run it where MLflow already is: chain it to training on the laptop.** For example a DVC pipeline (`dvc.yaml` stages `validate → train → register → promote`, run with `dvc repro`), or simply the end of the training script: register the new version and call the gate. Another option is a **self-hosted GitHub Actions runner** on the laptop, which can reach `127.0.0.1:5000`. Or an MLflow registry webhook / polling job that runs `promote.py` whenever a new model version is created.

### Question 12: Draw the pipeline you now have, from `git push` to a running container.

See [`lab/lab5.md`](lab/lab5.md): a Mermaid flowchart (GitHub renders it), plus a table of which steps are automatic, which need a human, and where the **code version** (the git commit SHA, which is also the image tag, decided when a commit lands on `main`) and the **model version** (the `champion` alias, decided by `promote.py`) are chosen. The two meet only at container startup, when `serve.py` loads `models:/food11@champion` inside the image `ghcr.io/hayamabdallatif664/mlops-lab-1:<sha>`.
