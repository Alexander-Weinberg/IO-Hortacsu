#############################
## IO2 PSet 2 Q1
#############################

## CTRL + L clears the workspace

using CSV, Random, Distributions, DataFrames, Kronecker, LinearAlgebra, Compat, Statistics, Plots, LaTeXStrings, Optim, LineSearches

df = CSV.read("ps2_ex1.csv", DataFrame)

d = Normal(0,1)

##########################################################
## Function to calculate prob_θ(x)
## x scalar
## n scalar
## β, ϕ, δ parameters
##########################################################

function prob_θ(x, n, β, ϕ, δ) 
    π_bar_n = x*β - ϕ - δ*log(n)
    π_bar_np1 = x*β - ϕ - δ*log(n+1)
    Φ_n = cdf(d, π_bar_n)
    Φ_np1 = cdf(d, π_bar_np1)
    prob_θ = Φ_n - Φ_np1
    return prob_θ
end

##########################################################
## Likelihood Function
## x vector
## n vector
## θ = [β, ϕ, δ] parameters
##########################################################

function likelihood(x, n, θ)
    β = θ[1]
    ϕ = θ[2]
    δ = θ[3]
    prob_θ_vec = prob_θ.(x, n, β, ϕ, δ) 
    log_prob_θ_vec = log.(prob_θ_vec)
    likelihood = -mean(log_prob_θ_vec) ## negative because minimizing
    if likelihood == Inf
        likelihood = 1e6
    end
    return likelihood
end

##########################################################
## Maximize Likelihood
##########################################################

θ_init = [.1, 1.0, .5]
res = optimize(θ -> likelihood(df.x, df.n, θ), θ_init)
θ_sol = Optim.minimizer(res)
lh = Optim.minimum(res)



    