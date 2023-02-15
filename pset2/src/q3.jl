
# ================================ #
# LOAD PACKAGES
# ================================ #
using Plots, CSV, CategoricalArrays, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics, GLM, Optim, BlackBoxOptim, LaTeXTabulars 

# PARAM
β                   = 0.999
# NUMPARAM
K                   = 20                # coarsness of state space
tol_u               = 1e12              # check utility bounded
tol_v               = 1e-10             # vfi tolerance
param_GUESS         = [0.0, 1.0, 5.0];
max_iter_v          = 100
# ================================ #
# DATA
# ================================ #
df                  = CSV.read("./input/ps2_ex3.csv", DataFrame)
T                   = length(df.milage)

##### RECOVER D 
df.d                = zeros(size(df,1))
df[2:end, "d"]      = diff(df.milage) .< 0      # 1 if change engine. 0 otherwise.

##### DISCRETIZE STATE SPACE 
describe(df.milage)                             # mostly between 34-98 miles
@assert any(isnan.(df.milage)) == 0             # none missing

# State space (The code below was written by GPT-3 !!!!)
xmin            = minimum(df.milage)
xmax            = maximum(df.milage) + 1
xgrid           = LinRange(xmin, xmax, K+1)

# Bin 
binned_data = [findall(x -> x <= b, xgrid)[end] for b in df.milage]
df.xbin     = binned_data

# State is the median mileage of the bin.
grouped_df = combine(groupby(df, :xbin), df -> median(df[!,:milage]))
rename!(grouped_df, :x1 => :x) # Rename the column

df         = innerjoin(df, grouped_df, on = :xbin)
states     = unique(df.x)

# ================================ #
# FLOW UTILITY
# ================================ #
function u(x, d, θ1, θ2, θ3,ϵ0,ϵ1)
    if d==0
        util = -θ1*x -θ2 * (x/100)^2 + ϵ0 
    elseif d==1
        util = -θ3 + ϵ1 
    else 
        @warn "Invalid decision entered"
    end

    if abs(util) > tol_u
        @warn "Utility unbounded"
    end

    return util
end

# ================================ #
# ESTIMATE TRANSITION MATRIX. D=0
# ================================ #

## Thank you again to GPT3 lol
frequency = zeros(K, K)             # transition matrix
for i = 1:(T-1)    
    # Skip if d=1
    if df.d[i+1] == 1.0
        continue 
    end

    # Sum frequency 
    start_index     = df.xbin[i]
    end_index       = df.xbin[i+1]
    frequency[end_index, start_index] += 1
end

# NORMALIZE TRANSITION MATRIX
P0       = frequency ./ sum(frequency,dims=1)

# Impose that if in last state, stay in last state if D=0
adjust   = zeros(K)
adjust[end] = 1
P0[:,end] = adjust
@assert isapprox(sum(P0,dims=1)', ones(K))

# ================================ #
# ESTIMATE TRANSITION MATRIX. D=1
# ================================ #
frequency = zeros(K, K)             # transition matrix
for i = 1:(T-1)    
    # Skip if d=0
    if df.d[i+1] == 0.0
        continue 
    end

    # Sum frequency 
    start_index     = df.xbin[i]
    end_index       = df.xbin[i+1]
    frequency[end_index, start_index] += 1
end

P1       = frequency ./ sum(frequency,dims=1)
# Impose that if in early state, plug in earlies seen state
max_col_nan = findlast(isnan.(sum(P1,dims=1)))[2]
adjust   = P1[:,max_col_nan+1]
P1[:,1:max_col_nan] .= adjust
@assert isapprox(sum(P1,dims=1)', ones(K))


# ================================ #
# SOLVE MODEL 
# ================================ #
x_k         = states 
function vfi(θ)
    
    # Initialize
    ev_init         = ones(K)    
    ev              = ev_init
    diff_v          = 1000              
    iter_v          = 0

    for iii in 1:max_iter_v
        iter_v      += 1
        f           = log.(exp(-θ[3] + β*ev[1]) .+ exp.(-θ[1].*x_k - θ[2].*(x_k ./ 100).^2 + β.*ev)) .+ 0.57
        ev_new      = P0 * f

        #____________________
        # CHECK BREAK UPDATE 
        diff_v      = maximum(abs.(ev - ev_new))
        if diff_v < tol_v
            break 
        end
        ev          = ev_new
    end
    if iter_v == max_iter_v
        @warn "Did not converge."
    else
        println("Solved in $iter_v iterations.")
    end
    return ev 
end

aaa = vfi(param_GUESS)

# ================================ #
# CCP 
# ================================ #
function ccp(θ,EV)
    p = exp.(-θ[3] + β*EV[1]) ./ (exp.(-θ[3] + β*EV[1]) .+ exp.(-θ[1].*x_k - θ[2].*(x_k ./ 100).^2 .+ β.* EV))
    
    return p
end

# ================================ #
# LIKELIHOOD FUNCTION 
# ================================ #
function neglikelihood(θ)
    # Compute value
    EV = vfi(θ)
    
    # Compute conditional choice probabilities across states
    c = ccp(θ,EV)
    
    # Map conditional choice probabilities to data
    T  = length(df.milage)

    p   .= c[df.xbin]

    d = vec(df[!,:d])

    # Return negative log-likelihood to minimize
    ll = sum(log.(p.^d .* (1 .- p).^(1 .- d)))
    
    return -ll
end

# neglikelihood(param_GUESS,df)

# ================================ #
# MLE
# ================================ #
o = optimize(neglikelihood, param_GUESS, NelderMead())
θ = o.minimizer
println(θ)