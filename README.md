
# Optimization Algorithms Project

This repository contains a collection of optimization algorithms implemented in Python. The project is structured into three phases, progressing from single-variable optimization to multi-variable unconstrained optimization, and finally to constrained optimization.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Phase 1: Single-Variable Optimization](#phase-1-single-variable-optimization)
- [Phase 2: Unconstrained Multi-Variable Optimization](#phase-2-unconstrained-multi-variable-optimization)
- [Phase 3: Constrained Optimization](#phase-3-constrained-optimization)

## Prerequisites

To run the scripts in this project, you need Python installed along with the `numpy` and `matplotlib` libraries.

To install the dependencies using pip:
```bash
pip install numpy matplotlib
````

## Project Structure

```
.
├── .gitignore
├── README.md
├── Phase 1/
│   ├── Phase1.py           # Single-variable optimization script
│   └── input.txt           # Input parameters for Phase 1 (required by script)
├── Phase 2/
│   ├── Phase2.py           # Powell's Conjugate Direction Method script
│   ├── Phase1Methods.py    # Helper module containing 1D search methods
│   ├── input.txt           # Input parameters for Phase 2
│   └── __pycache__/
└── Phase 3/
    ├── phase3.py           # Penalty Method for Constrained Optimization
    ├── phase2.py           # Powell's Method adapted for Phase 3
    └── __pycache__/
```

-----

## Phase 1: Single-Variable Optimization

This phase implements fundamental 1D search algorithms to find the minimum of a function within a specified interval.

### Algorithms

  * **Bounding Phase Method:** Used to bracket the minimum.
  * **Golden Section Search:** Refines the interval to find the minimum with high accuracy.

### Usage

To run the single-variable optimization script:

1.  Ensure `input.txt` is present in the `Phase 1/` directory.
2.  Execute the script:
    ```bash
    python "Phase 1/Phase1.py"
    ```

**Input Format (`input.txt`):**
To configure the input parameters, use the following format in `Phase 1/input.txt`:

```text
<lower_bound> <upper_bound> <delta> <initial_guess> <accuracy>
```

*Note: If `initial_guess` is not provided, the script will select a random one.*

**Output:**
The script generates the following log files:

  * `BoundingPhaseiteration.txt`
  * `Golden_Section_Search_Mehtod_iterations.txt`

-----

## Phase 2: Unconstrained Multi-Variable Optimization

This phase extends optimization to $N$-dimensions using **Powell's Conjugate Direction Method**. It utilizes the 1D search methods from Phase 1 to perform unidirectional searches along conjugate directions.

### Algorithms

  * **Powell's Conjugate Direction Method**
  * **Unidirectional Search:** Combines Bounding Phase and Golden Section methods.

### Usage

To execute the unconstrained optimization:

1.  Configure the parameters in `Phase 2/input.txt`.
2.  Run the script:
    ```bash
    python "Phase 2/Phase2.py"
    ```

**Input Format (`input.txt`):**
To define the function and bounds, use the following format in `Phase 2/input.txt`:

```text
<function_index>
<dimensions>
<lower_bound>,<upper_bound>
```

**Supported Functions (by index):**

1.  Sum of Squares
2.  Rosenbrock
3.  Dixon-Price
4.  Trid
5.  Zakharov

**Output:**

  * **Results:** Written to `output.txt`.
  * **Visualization:** A search path plot is displayed if dimensions $d \le 2$.

-----

## Phase 3: Constrained Optimization

This phase solves constrained optimization problems using the **Bracket-Operator Penalty Method**. It transforms constrained problems into unconstrained ones by adding penalty terms, which are then solved using the Powell's method implementation from Phase 2.

### Algorithms

  * **Bracket-Operator Penalty Method:** Handles inequality ($g(x) \le 0$) and equality constraints.
  * **Unconstrained Solver:** Powell's Conjugate Direction Method.

### Usage

To perform constrained optimization:

```bash
python "Phase 3/phase3.py"
```

*Note: This script uses hardcoded test problems (Problem 1, 2, etc.) with specific objective functions ($f$) and constraints ($g$)*.

**Features:**

  * Automatically adjusts the penalty parameter $R$.
  * Performs multiple test runs with random initial guesses.
  * Calculates statistics (Best, Worst, Mean, Std Dev) for the solutions found.

**Output:**
To view the results, check the generated `output.txt` file, which includes:

  * Parameter settings.
  * Convergence data for each run.
  * Final statistics.

