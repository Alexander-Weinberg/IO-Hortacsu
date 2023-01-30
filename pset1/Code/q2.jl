# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics

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
x_simple = x_simple[1:J,2:4]
x_simple = Matrix(x_simple)

# ================================ #
# LOG-LIKELIHOOD FUNCTION
# ================================ #
function loglike(θ)
    δ = θ[1:J]                      # J x 1 vector 
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
        end
        a           = di * Γ * transpose(x_simple)
        b           = δ .+  transpose(a)
        thirdbit    = -log(sum(exp.(b)))

        # updte likelihood
        likelihood += firstbit 
        likelihood += secondbit[1] 
        likelihood -= thirdbit 
    end
    return likelihood
end

function negloglike(θ)
    ll = -loglike(θ)
    return ll
end 


# # this works
theta_guess = zeros(J+ (L*K))
# negloglike(theta_guess)

# OPTIMIZE
Sol = optimize(negloglike, theta_guess, Optim.Options(iterations = 1)) # warmup
Sol = optimize(negloglike, theta_guess) # solve
Sol
