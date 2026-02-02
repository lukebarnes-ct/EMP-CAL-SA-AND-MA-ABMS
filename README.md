# Empirical Calibration of Single-Asset and Multi-Asset Agent-Based Models for Financial Markets

The Julia code used to implement the ABMs, in conjunction with the calibration and simulation methodology, for the Empirical Calibration of Single-Asset and Multi-Asset Agent-Based Models for Financial Markets dissertation by Luke Barnes is available in this Github repository. The repository contains all the Julia code used to process and clean the empirical data, calibrate and simulate each of the ABMs, perform a sensitivity analysis on each model variation's parameters, and generate and plot key model outputs.

## Repository Structure:

Data contains the empirical financial market data used and the obtained model output from both the calibration and simulation procedures for all models. 

Plots contains the specific visual output, used in the dissertation, of each of the model simulations and the conducted sensitivity analysis.

The following paragraphs acknowledges the Julia scrips used:

calculate-index-returns.jl, import_save_JSETOP40.jl, import_save_SSE50.jl and import_save_BSESN.jl are used in the empirical data cleaning and formulation process.

model-calibration-and-simulation-hl-abm.jl, model-calibration-and-simulation-fw-abm.jl and model-calibration-and-simulation-xu-abm.jl calibrate each of the model variations to find the optimal parameters for each index and generates simulations based on the found optimal parameters. 

statistics-and-plotting-hl-abm.jl, statistics-and-plotting-fw-abm.jl and statistics-and-plotting-xu-abm.jl are used to create the table and figure output of the simulation results shown in the dissertation. 
