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