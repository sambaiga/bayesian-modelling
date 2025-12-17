## Setting Up the Project (Basic)

Follow these steps to quickly install all dependencies and set up the working environment.

1. **Clone the repository**:

  ```bash
   git clone https://github.com/sambaiga/bayesian-modelling.git
  ```

2. **Install** [uv](https://docs.astral.sh/uv/getting-started/installation/) (if not already installed):

  ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create and activate the virtual environment**

  ```bash
    # Create the environment in .venv
    uv venv --python 3.11
  ```

  ```bash
    # Activate the environment # macOS/Linux
    source .venv/bin/activate
  ```

  OR

  ```bash
     .venv\Scripts\activate     #Windows PowerShell
  ```

4. **Install all dependencies**

This command reads the ``pyproject.toml`` file and installs all required packages (main, dev, test, etc.).

 ```bash
    uv sync --all-extras --dev
 ```

Then  **verify everything works**

  ```bash  
  uv run python -c "import bayes; print('Bayes imported successfully!')"
  ```

## 💻 Developer Setup (Optional)

5. **Install the project in editable mode**.
   This is crucial for development and ensures you can import your local ark package (e.g., in Jupyter notebooks).

    ```bash
        uv pip install -e .
    ```

6. **Initialize pre-commit hooks**
This sets up hooks that automatically format and lint your code before each commit. Since pre-commit is now installed in your environment, we run it via `uv run`.

    ```bash
    uv run pre-commit install
    ```

  then

  ```bash
    uv run pre-commit autoupdate
  ```

7. **Enable nbdime** for improved Jupyter Notebook version diffs

 ```bash
 uv run nbdime config-git --enable --global
```

8. **Install git-cliff (for changelog)**
   If you want to maintain an automated changelog:
   If you want to maintain an automated changelog:

  ```bash
    # macOS
    brew install git-cliff

    # Linux
    curl -LsSf https://github.com/orhun/git-cliff/releases/latest/download/git-cliff-install.sh | sh

    # Windows (PowerShell)
    winget install -e --id gitcliff.gitcliff
    # or
    iwr https://github.com/orhun/git-cliff/releases/latest/download/git-cliff-install.ps1 -useb | iex

    # Any OS (if you have Rust/Cargo)
    cargo install git-cliff
  ```

</details>

## 🏃 Running the Project

Once setup is complete:

1. Open the project in **VS Code**. Make sure you have the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter Notebook](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions installed.
2. Run the **appropriate** notebook (file with `.ipynb` extension) from the `/notebooks` folder.
3. Verify imports and paths load correctly.
