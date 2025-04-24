# Structural Bias Analysis in the Artificial Bee Colony Algorithm

This repository contains the implementation for the study **"Comprehensive Analysis on Structural Bias in the Artificial Bee Colony Algorithm"**. The project investigates **structural bias** in the **Artificial Bee Colony (ABC)** algorithm, comparing its **Signature Factor (SSF)** with four other metaheuristic algorithms: **Particle Swarm Optimization (PSO)**, **Differential Evolution (DE)**, **Genetic Algorithm (GA)**, and **Random Search with Greedy Selection (RS-GS)**. 

The code implements the **Generalized Signature Test (GST)** to quantify structural bias and generates a high-quality plot comparing SSF over 100 iterations, saved as `output.png` (100 DPI). Findings show that ABC exhibits minor central bias, primarily in the initial iterations, with minimal impact from population size and slight amplification in higher dimensions.

## Features
      - Implements PSO, ABC, DE, GA, and RS-GS with consistent parameters:
        - Population size (N): 100 (change as per the rule, see the article)
        - Dimensions (D): 2 (change as per the rule, see the article)
        - Iterations: 100
      - Computes SSF to quantify population sparsity (high SSF indicates clustering, low SSF indicates diversity).
      - Generates a high-resolution plot with distinct colors and markers for each algorithm.
      - Saves result as PNG for versatile use.
      - Validates findings from the paper, focusing on ABC’s minor central bias.
      

## Prerequisites
- **Python 3.6+**
- Required libraries:
  - `numpy`: For numerical computations.
  - `matplotlib`: For plotting.

Install dependencies using pip:

      pip install numpy matplotlib

## Installation
1. Clone the repository:

        git clone https://github.com/kanchan999/SB_ABC.git
        cd SB_ABC

3. Install dependencies (see Prerequisites).

## Usage
1. Run the script:

        python SB_ABC.py

2. The script will:
      Execute PSO, ABC, DE, GA, and RS.
      Compute SSF for each iteration using GST.
      Generate a plot comparing SSF across algorithms.
      Save the plot as output.png (high-resolution).

3. Check the generated files (output.png, output.svg) in the project directory.

## Expected Output
      Plot: A high-quality figure showing SSF ((\eta)) over 100 iterations:
        PSO: High SSF (e.g., 0.4 to 0.9), indicating strong convergence and clustering.
        ABC: Moderate SSF (e.g., 0.36 to 0.54), with minor central bias in early iterations.
        DE: Steady SSF increase, balancing exploration and convergence.
        GA: Moderate SSF, with mutation preserving diversity.
        RS-GS: Low SSF (e.g., 0.3 to 0.5), reflecting high diversity from random exploration.

      Files:
        output.png: High-resolution raster image (100 DPI).

## Project Structure
      SB_ABC/
      ├── .gitignore
      ├── LICENSE
      ├── README.md
      ├── opt_algorithms_ssf_comparison.py
      └── output.png  (generated after running the script)

## License
This project is licensed under the MIT License. See the  file for details.

## Contact
For questions or feedback, open an issue on GitHub or contact Kanchan Rajwar at kanchanrajwar1519@gmail.com.
      





