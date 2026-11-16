# Learning Menu-Based Mechanisms for Truthful Budget-Feasible Procurement

This repository contains an anonymized implementation for the paper **"Learning Menu-Based Mechanisms for Truthful Budget-Feasible Procurement"**.

> **Note:** This repository is strictly anonymized for double-blind review. Our framework introduces a novel neural mechanism design paradigm that utilizes a Transformer-based Base Menu Network (BFMNet) followed by an ex-post MILP transformation to guarantee exact Individual Rationality (IR), Dominant Strategy Incentive Compatibility (DSIC), and strict Budget Feasibility without relying on restrictive global discretization.

## Requirements

To install the necessary requirements, please ensure you have Python 3.8+ installed, and run:
```bash
pip install torch numpy tqdm gurobipy
```

**Important License Note:** The Ex-Post Transformation module (`MILP.py`) requires a valid Gurobi Optimizer license. Academic licenses are available for free on the [Gurobi website](https://www.gurobi.com/).

## Training

The codebase is organized by experimental settings (e.g., `Additive/`, `Monotone_Submodular/`, `Asymmetry/`, `Large_Scale/`). 

To train the BFMNet models in the paper, navigate to your desired setting directory and follow this two-step procedure:

**1. Generate Datasets:**
First, generate the training, validation, and final test datasets:
```bash
cd Additive/
python generate_data.py
```
*This script will create a `data/` folder containing the initialized bids, values, and budgets.*

**2. Train the BFMNet:**
Run the training script corresponding to the number of agents n (e.g., 8 bidders). 
```bash
python 8.py --data_dir ./data/bidder=8 --n_agents 8 --d_emb 3 --n_layer 3 --n_head 8
```
You can adjust hyperparameters such as `--d_emb` (embedding dimensions), `--n_layer` (transformer layers), and `--n_head` (attention heads) directly via the command line.

## Evaluation

To evaluate the mechanism and enforce the strict economic guarantees (exact IR, DSIC, and Budget Feasibility), we run the Ex-Post Transformation module using MILP.

Run the following command to evaluate the pre-generated menus on the final test set:
```bash
python MILP.py --data_dir ./data/bidder=8 --n_bidder 8 --test_set final_test
```

This script will dynamically adjust the base menus, calculate the final allocations/payments, and output the comprehensive evaluation metrics including the total utility, expected payment, and the empirical discretization rate.