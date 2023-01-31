# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, DataFrames, GLM, Optim, Random, LinearAlgebra, LaTeXTabulars, Statistics

# ================================ #
# IMPORT DATA
# ================================ #
df      = CSV.read("./input/ps1_ex2.csv", DataFrame)

mat     = Matrix(df)
jvec    = convert(Vector{Integer}, mat[:,1])
dmat    = mat[:,5:6]
xmat    = mat[:,2:4]

# PARAM
J       = 30            # includes outside option
I       = 4000
L       = 2
K       = 3

# grab unique x's for summation. there has got to be simpler way
x_simple = unique(df, [:choice])
x_simple = sort!(x_simple, [:choice])
x_simple = x_simple[1:J+1,2:4]
x_simple = Matrix(x_simple)

# ================================ #
# LOG-LIKELIHOOD FUNCTION
# ================================ #
function loglike(θ)
    δ = θ[1:J]                      # J x 1 vector 
    δ = [δ;1]                       # append outside option
    g = θ[J+1:end]                  
    Γ = reshape(g, 2, 3)            # L x K matrix 


    likelihood      = 0             # initialize
    for i in 1:I
        choice      = jvec[i]
        di          = reshape(dmat[i,:],1,2)
        xj          = xmat[i,:]

        # compute for obs i.
        if choice == (J+1)
            firstbit    = 0
            secondbit   = 0
        else 
            firstbit    = δ[choice] 
            secondbit   = di * Γ * xj
            secondbit   = secondbit[1]
        end

        thirdbit    = -log(sum(exp.(δ + (dmat[i, :]' * Γ * x_simple')')))

        # updte likelihood
        likelihood += firstbit 
        likelihood += secondbit
        likelihood += thirdbit 

        # 

        # # Jun and Max
        # likelihood += 



    end
    return likelihood
end

function negloglike(θ)
    ll = -loglike(θ)
    return ll
end 


# # this works
theta_guess = ones(J + (L*K))

# OPTIMIZE
Sol = optimize(negloglike, theta_guess, Optim.Options(iterations = 1)) # warmup
Sol = optimize(negloglike, theta_guess, LBFGS(), autodiff=:forward) # solve

# SOLUTION
θ_answer = Sol.minimizer
δ_answer = [θ_answer[1:J];1]
Γ_answer =  reshape(θ_answer[J+1:end], 2, 3)            # L x K matrix 


# delta
delta = hcat("\$delta_{" .* string.(1:J+1) .* "}\$", round.(δ_answer, digits = 3))
latex_tabular("./Output/ps1_q2_deltas.tex",
              Tabular("cc"),
              [Rule(:top),
               delta,
               Rule(:bottom)])

# Gamma
gamma = hcat("\$gamma_{" .* ["11", "21", "12", "22", "13", "23"] .* "}\$", round.(θ_answer[J+1:end], digits = 3))
latex_tabular("./Output/ps1_q2_gammas.tex",
              Tabular("cc"),
              [Rule(:top),
              gamma,
               Rule(:bottom)])


# REGRESSION

# Build dataframe
data = unique(df, [:choice])
data.delta = δ_answer
rename!(data,[:choice,:x1, :x2, :x3, :d1, :d2, :delta])


# Regression
ols = lm(@formula(delta ~ -1 + x1 + x2 + x3), data)

# Store estimates
ξ = data.delta .- predict(ols);
β = coef(ols);
   
beta = hcat("\$\beta_{" .* string.(1:3) .* "}\$", round.(β, digits = 3))
latex_tabular("output/ps1_q2_betas.tex",
              Tabular("cc"),
              [Rule(:top),
               beta,
               Rule(:bottom)])