# NuQu
Physics 765, S26: Quantum Algorithms and Error Correction. Final Project: Fault-tolerant resource estimation for QPE for Dynamical Pion Effective Field Theory (EFT) using Qubitized Walk operators as the subroutine for Hamiltonian evolution e^-iHt. We construct the Hamiltonian using OpenFermion for  lattice dimension 1, 2, or 3, number of lattice sites per side (square/cubic lattice), and system parameters, then use pyLIQTR to construct a Qubitized Walk operator and evaluate its resource costs. Estimates are compared to similar results using a Trotterized time evolution operator from https://doi.org/10.48550/arXiv.2312.05344.
This is a first step in creating a reproducible, full-stack pipeline for exploring how fault-tolerant quantum computers could be used for realistic nuclear simulations.

## Installation

### Prerequisites
* **Python 3.10.x** (Required). This project uses dependencies not yet compatible with Python 3.12+.


### Setup
1. Clone the repository:
   git clone [https://github.com/brandonnfriend/NuQu.git](https://github.com/brandonnfriend/NuQu.git)
   cd NuQu
2. Create and activate a virtual environment:
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install dependencies:
    pip install -r requirements.txt

## Reproducing Results
To generate the primary analysis and plots presented in the final report, run the main execution scripts: run_nucleon_sweep.py (your choice of input parameters, especially lattice sites per side L) and plot_sweep_data.py which plots the data. In plot_data_sweep.py, update the filepath based on the save location of the data from run_nucleon_sweep.py, and choose your plotting function.

## Output Data
* Data Dumps: Raw output is automatically saved to the data/ directory, organized by date (e.g., data/2026-04-21/).

* Visualizations: Plots are rendered to the data/(today's date) folder as .png files.

## Methods
In preparation

## Status
This is a work in progress for an upcoming publication. Code will be optimized and new features and analysis will be implemented. 

## Citation
In preparation

## Contact
github: brandonnfriend

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
