
# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, CategoricalArrays, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics, GLM, Optim, BlackBoxOptim, LaTeXTabulars 

# PARAM

# NUMPARAM
K                   = 10                # coarsness of state space
tol_u               = 1e12              # check utility bounded

# ================================ #
# DATA
# ================================ #
df                  = CSV.read("./input/ps2_ex3.csv", DataFrame)

##### RECOVER D 
df.d                = zeros(size(df,1))
df[2:end, "d"]      = diff(df.milage) .< 0 # one if change engine. 0 otherwise.

##### DISCRETIZE STATE SPACE 
describe(df.milage)                         # mostly between 34-98 miles
@assert any(isnan.(df.milage)) == 0         # none missing

# State space (The code below was written by GPT-3 !!!!)
xmin            = minimum(df.milage)
xmax            = maximum(df.milage) + 1
xgrid           = LinRange(xmin, xmax, K+1)

# Bin 
binned_data = [findall(x -> x <= b, xgrid)[end] for b in df.milage]
df.xbin     = binned_data
df[!,'x']        = combine(, :milage => mean)

grouped_df = combine(groupby(df, :xbin), df -> mean(df[:milage]))

# Rename the column
rename!(grouped_df, :x1 => :mean_value)

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
# DATA PREP
# ================================ #


# ================================ #
# SOLVE FOR delta_jt
# ================================ #



# ================================ #
# LOOP OVER MARKETS
# ================================ #
function do_delta_loop(df, Gamma, delta_init, verbose=false)

    delt_mat = zeros(T,J)

    for t in 1:T
        if verbose
            # market index
            println("---\nNow solving market $t/$T")
        end
        market_idx = (df[:,1] .== t)

        # Get the data
        s_jt_obs = df[market_idx, 3]

        p_jt = df[market_idx, 4]
        x_jt = df[market_idx, 5
        xtilde_jt = [p_jt x_jt]

        # solve for delta_jt
        delt_mat[t, :] = transpose(solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma, verbose))
    end
    delt_vec = vec(transpose(delt_mat))
    return delt_vec
end

#########################################
# Function to compute structural errors
#########################################
function compute_residuals(data)

    # Regression 2sls 
    firststage      = lm(@formula(p ~  x + z1 + z2 + z3 + z4 + z5 + z6), data)
    p_hat           = predict(firststage)
    data[!,"phat"]  = p_hat
    
    ols             = lm(@formula(delta_jt ~  phat + x), data)
    beta            = GLM.coef(ols)
    delta_jt        = data.delta_jt
    x               = data.x
    p               = data.p
    ξ_jt            = delta_jt  .-beta[1] - beta[2].*p - beta[3].*x
   
    return ξ_jt, beta
end

################################################
# Function to calculate GMM objective function
################################################
function GMM_objective(Gamma_vec)
    println("Now evaluating obj function.")
    # Build Gamma matrix
    Gamma_11 = Gamma_vec[1]
    Gamma_21 = Gamma_vec[2]
    Gamma_22 = Gamma_vec[3]
    Gamma = [Gamma_11 0.0; Gamma_21 Gamma_22]

    # Get deltas
    data                    = df[:,4:11]
    delta_jt                = do_delta_loop(df,Gamma,delta_init)
    data[!,"delta_jt"]      = delta_jt

    # compute residuals
    one_jt                  = ones(J*T)
    df.ones                 = one_jt
    Z                       = hcat(one_jt, Matrix(df[:,5:11]))
    # Z                       =  Matrix(df[:,6:11])

    # compute GMM error
    ξ_jt, beta              = compute_residuals(data)

    Ω_inv                   = inv(transpose(Z)*Z)                         # homoskedasticity
    objective               = (transpose(ξ_jt)*Z) * Ω_inv * (transpose(Z)*ξ_jt) 
    # objective               = (transpose(ξ_jt)*Z)  * (transpose(Z)*ξ_jt) 
    println("Val = $objective")

    # robust 
    if isnan(objective)
        objective = 1e10
    end

    return objective, beta, delta_jt
end

################################################
# Function for non-linear solver to optimize
################################################
function aux_gmm_obj(Gamma_vec)
    objective, beta, delta_jt   = GMM_objective(Gamma_vec)
    return objective
end

################################################
# Function to compute cross-price elasticities
################################################
function compute_cprice_elasticity(alpha, s_jt, δ_jt, δ_kt, xtilde_jt, xtilde_kt, Gamma_vec)
    Gamma_11 = Gamma_vec[1]
    Gamma_21 = Gamma_vec[2]
    Gamma_22 = Gamma_vec[3]
    Gamma = [Gamma_11 0.0; Gamma_21 Gamma_22]
    integrand_ijk = ones(Ns,1)
    integrand_jk = 0
    p_kt = xtilde_kt[1]
    for i in 1:Ns
        # Draw shock i
        v_i = v_mat[:,i]
        
        # Compute individual shares 
        share_ijt = compute_share_ijt(δ_jt, xtilde_jt, Gamma, v_i)
        share_ijt = share_ijt[1]
        share_ikt = compute_share_ijt(δ_kt, xtilde_kt, Gamma, v_i)
        share_ikt = share_ikt[1]

        # Compute elasticity integrals nonparametrically
        integrand_ijk[i,1] = alpha*p_kt*share_ijt*share_ikt
        integrand_jk = mean(integrand_ijk, dims = 1)
        integrand_jk = alpha*p_kt*integrand_jk[1]/s_jt
    end
    return integrand_jk
end

################################################
# Function to compute own-price elasticities
################################################
function compute_oprice_elasticity(alpha, s_jt, δ_jt, xtilde_jt, Gamma_vec)
    integrand_ijj = ones(Ns, 1)
    integrand_jj = 0
    Gamma_11 = Gamma_vec[1]
    Gamma_21 = Gamma_vec[2]
    Gamma_22 = Gamma_vec[3]
    Gamma = [Gamma_11 0.0; Gamma_21 Gamma_22]
    p_jt = xtilde_jt[1]
    for i in 1:Ns
        # Draw shock i
        v_i = v_mat[:,i]
        
        # Compute individual shares 
        share_ijt = compute_share_ijt(δ_jt, xtilde_jt, Gamma, v_i)
        share_ijt = share_ijt[1]

        # Compute elasticity integrals nonparametrically
        integrand_ijj[i,1] = share_ijt*(1-share_ijt)
        integrand_jj = mean(integrand_ijj, dims = 1)
        integrand_jj = -1*alpha*p_jt*integrand_jj[1]/s_jt   
    end
    return integrand_jj[1]
end

################################################
# Computes JxJ elasticity matrix
################################################
function compute_price_elasticity_matrix(alpha, s_jt, δ_jt, xtilde_jt, Gamma_vec)
    elasticities = ones(J,J)
    for j in 1:J
        for k in 1:J
            if j != k 
                elasticities[j,k] = compute_cprice_elasticity(alpha, s_jt[j], δ_jt[j], δ_jt[k], reshape(xtilde_jt[j,1:2],1,2), reshape(xtilde_jt[k,1:2],1,2), Gamma_vec) 
            else 
                elasticities[j,k] = compute_oprice_elasticity(alpha, s_jt[j], δ_jt[j], reshape(xtilde_jt[j,1:2],1,2), Gamma_vec)
            end
        end
    end
    return elasticities    
end

################################################
# Computes elasticity matrix for all markets
################################################
function loop_price_elasticity(alpha,df,Gamma_vec)
    Gamma_11 = Gamma_vec[1]
    Gamma_21 = Gamma_vec[2]
    Gamma_22 = Gamma_vec[3]
    Gamma = [Gamma_11 0.0; Gamma_21 Gamma_22]
    δ_jt                                = do_delta_loop(df,Gamma,delta_init)
    df.elasticity1                      = ones(T*J)
    df.elasticity2                      = ones(T*J)
    df.elasticity3                      = ones(T*J)
    df.elasticity4                      = ones(T*J)
    df.elasticity5                      = ones(T*J)
    df.elasticity6                      = ones(T*J)
    df[!,"δ_jt"]                        = δ_jt
    for t in 1:T 
        market_idx = (df[:,1] .== t)
        p_jt = df[market_idx, 4]
        x_jt = df[market_idx, 5]
        xtilde_jt = [p_jt x_jt]
        s_jt = df[market_idx, 3]
        elas_mat = compute_price_elasticity_matrix(alpha, s_jt, δ_jt, xtilde_jt, Gamma_vec)
        df[market_idx,"elasticity1"] = Vector(elas_mat[:,1]) 
        df[market_idx,"elasticity2"] = Vector(elas_mat[:,2]) 
        df[market_idx,"elasticity3"] = Vector(elas_mat[:,3]) 
        df[market_idx,"elasticity4"] = Vector(elas_mat[:,4]) 
        df[market_idx,"elasticity5"] = Vector(elas_mat[:,5]) 
        df[market_idx,"elasticity6"] = Vector(elas_mat[:,6]) 
    end
    return df
end

################################################
## Perform outer loop using nonlinear solver
################################################
Gamma_init          = [1.0,1.0,1.0]  
Sol                 = optimize(aux_gmm_obj, Gamma_init)
gamma_answer        = Optim.minimizer(Sol)
val, beta, delta_jt = GMM_objective(gamma_answer)

################################################
## Generate matrix of average elasticities
################################################
Gamma_vec = gamma_answer
alpha = -beta[2]
df_new = loop_price_elasticity(alpha,df,Gamma_vec)
df_elasticity = hcat(df_new[:,1:2],df_new[:,13:18])
elasticity = combine(groupby(df_elasticity, ["choice"]), 
    df -> DataFrame(e1 = mean(df.elasticity1)),
    df -> DataFrame(e2 = mean(df.elasticity2)),
    df -> DataFrame(e3 = mean(df.elasticity3)),
    df -> DataFrame(e4 = mean(df.elasticity4)),
    df -> DataFrame(e5 = mean(df.elasticity5)),
    df -> DataFrame(e6 = mean(df.elasticity6))
)
elasticity = round.(elasticity, digits = 4)

################################################
# Output elasticity matrix to LaTeX
################################################
labels = ["Product 1", "Product 2", "Product 3", "Product 4", "Product 5" ,"Product 6"]
elasticity.labels = labels
elasticity = elasticity[!,[8,2,3,4,5,6,7]]
elasticity_mat = Matrix(elasticity)

latex_tabular("./Output/blp_elasticities.tex",
    Tabular(raw"lcccccc"),
    [Rule(:top),
    ["", "Product 1", "Product 2", "Product 3", "Product 4", "Product 5", "Product 6"],
    Rule(:mid),
    elasticity_mat,
    Rule(:bottom)]
)









