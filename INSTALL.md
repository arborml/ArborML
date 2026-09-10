# ArborML 1.0 — Installation

ArborML is a cross-platform desktop application written in  Python. It requires
**Python 3.9 or newer** and runs on CPU only — no GPU is needed. 8 GB of ram was enough.

**Tested configurations**

| Platform | Architecture | Python |
|---|---|---|
| macOS 15 (MacBook Air, M1, 16 GB RAM) | ARM64 | 3.9.10 |
| macOS 15 (iMac 2019, 32 GB RAM) | x86-64 | 3.9.10 |
| Windows 11 Pro 25H2 (Dell Precision 7875 Tower, 128 GB RAM) | x86-64 | 3.14.3 |
| Ubuntu 26.04.1 LTS (HP Z230 Tower Workstation, 8 GB RAM) | x86-64 | 3.14.4 |

## 0. Get the code
```bash
git clone https://github.com/arborml/ArborML.git
cd ArborML
---
## 1. System packages (once per machine)
### macOS
ArborML needs Tcl/Tk for its graphical interface and OpenMP for XGBoost.
```bash
# OpenMP — REQUIRED if you want to use XGBoost or SHAP.
# Without it, "import xgboost" fails and the corresponding options
# are disabled in the interface.
brew install libomp
# Tcl/Tk — only needed if you use Homebrew's Python.
# The official installer from python.org already bundles Tcl/Tk.
# With pyenv, install tcl-tk first and rebuild your Python version.
brew install python-tk
```
Homebrew can be installed from https://brew.sh.

### Windows

- Install Python 3.9 or newer from https://www.python.org/downloads/.
  **Keep the “tcl/tk and IDLE” option checked** during installation.
- No other system package is required.

### Ubuntu / Debian *(untested)*

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-tk python3-dev libgomp1
# Optional: better rendering of Chinese, Japanese and Korean text
sudo apt install -y fonts-noto-cjk
```
---
## 2. Create a virtual environment (recommended)

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Windows — PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, allow it for the current
session only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
### Windows — cmd.exe
```bat
python -m venv .venv
.venv\Scripts\activate
```
## 3. Install Python packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## 4. Run
```bash
python ArborML.py
```
---
## Contact
tarik.sadat@uphf.fr
