# Lab 5 - The pipeline from `git push` to a running container

Legend: 🤖 = automatic, 👤 = needs a human. The two places where a version is decided are marked **CODE VERSION** and **MODEL VERSION**.

```mermaid
flowchart TD
    subgraph code["Code path (GitHub)"]
        A["👤 git push / open pull request"] --> B["🤖 GitHub Actions: job test<br/>uv sync --frozen, ruff check, pytest"]
        B -- pass --> C["🤖 job smoke<br/>docker build, run with SKIP_MODEL=1, curl /health"]
        B -- fail --> X["🤖 run red, PR blocked by branch protection"]
        C -- fail --> X
        C -- "pass, PR branch" --> D["👤 review + merge the PR<br/>(test and smoke must be green)"]
        D --> E
        C -- "pass, on main" --> E["🤖 job push<br/>build + push ghcr.io/hayamabdallatif664/mlops-lab-1:&lt;commit-sha&gt;"]
        E --> CV(["CODE VERSION = git commit SHA<br/>= image tag"])
    end

    subgraph model["Model path (laptop + MLflow)"]
        F["👤 dvc pull / dvc checkout (data version)"] --> G["👤 validate_data.py on the real data<br/>(exit 1 stops the pipeline)"]
        G --> H["👤 train.py on the laptop<br/>logs params, metrics, model to MLflow"]
        H --> I["👤 register the run's model as food11 vN"]
        I --> J["👤 promote.py --version N<br/>quality gate: acc >= champion + 2*se"]
        J -- pass --> K(["MODEL VERSION = alias champion -> vN"])
        J -- fail --> L["alias challenger -> vN, champion unchanged"]
    end

    CV --> M["👤 deploy: docker pull ghcr.io/...:&lt;sha&gt;<br/>docker compose up / restart inference"]
    K --> M
    M --> N["🤖 container startup: serve.py loads models:/food11@champion from MLflow"]
    N --> O(["Running container = image &lt;sha&gt; + model champion"])
```

## What is automatic and what still needs a human

| step | automatic? | where the version is decided |
|---|---|---|
| lint, unit tests, API test with a fake model (`test`) | 🤖 on every push and PR | |
| build the image and check it starts (`smoke`) | 🤖 after `test` passes | |
| merge to `main` | 👤 a reviewer, only possible when `test` and `smoke` are green (branch protection) | **code version**: the commit that lands on `main` |
| push the image to GHCR (`push`) | 🤖 only on `main` | the image tag **is** the commit SHA |
| get the data (`dvc pull`) and validate it (`validate_data.py`) | 👤 on the laptop (the runner has no access to the data) | the data version = `data.dvc` in that commit |
| train, register a version | 👤 on the laptop (no GPU and no data in CI) | |
| apply the quality gate (`promote.py`) | 👤 run by hand today; could be automatic if MLflow were reachable from CI (e.g. DagsHub's hosted MLflow) | **model version**: the `champion` alias |
| deploy (pull image, restart `inference`) | 👤 | the container combines image `<sha>` + model `@champion` at startup |

The image and the model are released independently. A new commit produces a new image without touching the model, and moving `champion` changes the served model without rebuilding the image (only a restart of `inference` is needed, Lab 4 Q8).
