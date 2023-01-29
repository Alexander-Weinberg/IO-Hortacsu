# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics

# ================================ #
# IMPORT DATA
# ================================ #
df      = CSV.read("./input/ps1_ex2.csv", DataFrame)


# ================================ #
# LOG-LIKELIHOOD FUNCTION
# ================================ #
function loglike(θ)
    δ = θ[1]                        # J x 1 vector 
    Γ = θ[2]                        # L x K matrix 

    
    firstbit = sum(y_ij * \delta      _j)

    \sum_i \sum_j y_{ij} \delta_j + \sum_i \sum_j y_{ij} d_i' \Gamma x_j - N \log \left( {\sum_{k=1}^{J+1} \exp(\delta_k + d_i' \Gamma x_k)} \right) 




end










