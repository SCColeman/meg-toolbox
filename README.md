# meg_toolbox

Robust and simple tools for MEG analysis, built around MNE-Python. 
Enables fieldtrip-style volume/grid-based beamformer workflows and aesthetic brain plots.

---

## Installation

### 1. Create a standard MNE-python conda environment

```bash
conda create --name mne --channel=conda-forge mne
conda activate mne
```

### 2. Install MNE-Connectivity

```bash
conda install -c conda-forge mne-connectivity
```

### 3. Clone meg-toolbox and go to directory

```bash
cd path/to/meg_toolbox
```

### 4. Install meg-toolbox

```bash
python -m pip install .
```

