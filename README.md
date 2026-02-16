# BIOENG 456: Controlling Behavior in Animals and Robots

Welcome to BIOENG 456: **Co**ntrolling **B**ehavior in **A**nimals and **R**obots (CoBAR)! In this course, we will program [NeuroMechFly 2.0](https://www.nature.com/articles/s41592-024-02497-y) (a digital twin of the adult fruit fly *Drosophila melanogaster*) to interact with the environment and perform complex behaviors.

## Getting started
To begin with the course materials, ensure you have Git, uv, FFmpeg, and Visual Studio Code installed.

### Installing Git
Git is essential for version control and collaboration. Download and install Git from [git-scm.com](https://git-scm.com/downloads).

### Installing uv
uv is a Python package manager that simplifies the installation of Python packages and their dependencies. See the [uv installation guide](https://docs.astral.sh/uv/#installation) for installation instructions.

### Installing FFmpeg
FFmpeg is used for encoding and saving videos of simulations. Check if FFmpeg is already installed by running:
```sh
ffmpeg -version
```
If FFmpeg is not installed, follow the instructions for your operating system:

**macOS** (using [Homebrew](https://brew.sh)):
```sh
brew install ffmpeg
```

**Windows** (using [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):
```sh
winget install --id Gyan.FFmpeg -e --source winget
```
After installation, restart your terminal for the `ffmpeg` command to become available.

**Linux (Debian/Ubuntu)**:
```sh
sudo apt update && sudo apt install ffmpeg
```

**Linux (Fedora)**:
```sh
sudo dnf install ffmpeg-free
```

Verify the installation by running:
```sh
ffmpeg -version
```

### Installing Visual Studio Code
- Download and install [Visual Studio Code](https://code.visualstudio.com)
- Open Visual Studio Code and navigate to the Extensions view by clicking on the Extensions icon in the Activity Bar on the left side of the window (or by pressing `Ctrl+Shift+X` on Windows/Linux and `Cmd+Shift+X` on Mac).
- Search for "Python" and install the Python extension provided by Microsoft.
- Search for "Jupyter" and install the Jupyter extension provided by Microsoft.

### Cloning the repository and setting up the environment
```sh
git clone https://github.com/NeLy-EPFL/cobar-2026.git
cd cobar-2026
uv sync
```

### Running the Jupyter Notebooks
Open the `cobar-2026` folder with Visual Studio Code: **File > Open...**

Open the Explorer view in Visual Studio Code by clicking on the Explorer icon in the Activity Bar on the left side of the window (or by pressing `Ctrl+Shift+E` on Windows/Linux and `Cmd+Shift+E` on Mac).

Open one of the .ipynb files within the Explorer view (e.g., `notebooks/week1/kinematic_replay.ipynb`)

Change the kernel to `flygym` (for how to change kernel, refer to https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)

For more instructions on how to work with Jupyter Notebooks in Visual Studio Code, refer to https://code.visualstudio.com/docs/datascience/jupyter-notebooks.

## Updating the repository
New materials will be released every week. Update the repository by:
```sh
cd cobar-2026
git pull
```

## Special notes for rendering on machines without a display
If you are using a machine without a display (e.g., a server), you will need to change the renderer to EGL. This requires setting the following environment variables before running FlyGym. We recommend adding these lines to the `.venv/bin/activate` file to ensure they are set every time you activate the virtual environment:
```sh
echo 'export MUJOCO_GL=egl' >> .venv/bin/activate
echo 'export PYOPENGL_PLATFORM=egl' >> .venv/bin/activate
```