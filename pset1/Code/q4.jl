# cd("/Users/Monica/Documents/UChicago/Classes/Industrial Organization II")
cd("/Users/Monica/Downloads/IO-Hortacsu-main/pset1")

# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics, Econometrics, Optim, BlackBoxOptim

# PARAM
T = 100
J = 6
Ns = 50

# NUMPARAM
delta_init      = ones(J,1) ./ J
tol_delta       = 1e-14
max_iter_delta  = 1000

Random.seed!(04211996)


# ================================ #
# IMPORT DATA
# ================================ #
df = CSV.read("./input/ps1_ex4.csv", DataFrame)

# ================================ #
# INDIVIDUAL SHARE PREDICTION 
# ================================ #
function compute_share_ijt(δ_jt, xtilde_jt, Gamma, v_i)
    # = Given shocks v_i, average product utility δ_jt, product characteristics x,p, and parameter Gamma
    # This function computes share_ijt =#

    # δ_jt is avg. utility for product j in market t
    # x_jt is product characteristics
    # Γ is the correlation matrix
    # v_l is the individual preference shock
    inner           = xtilde_jt * Gamma * v_i 
    top             = exp.(δ_jt .+ inner)
    bottom          = sum(top) + 1
    share_ijt       = top / bottom
    return share_ijt
end

# ================================ #
# MARKET SHARE PREDICTION 
# ================================ #
function compute_share_jt(δ_jt, xtilde_jt, Gamma)

    # initialize
    share_ijt = zeros(Ns,J)
    share_jt  = zeros(1,J)

    # draw shocks
    v_mat = randn(2,Ns)
    v_mat = zeros(2,Ns)

    for i in 1:Ns
        # Draw shock i
        v_i = v_mat[:,i]
        
        # Compute individual shares 
        output = compute_share_ijt(δ_jt, xtilde_jt, Gamma, v_i)
        share_ijt[i,:] = transpose(output)
    end

    # compute share
    share_jt = sum(share_ijt, dims=1) ./ Ns

    return share_jt
end

# ================================ #
# SOLVE FOR delta_jt
# ================================ #
function update_delta(delta_old_jt, s_jt_obs, xtilde_jt, Gamma)

    # Given delta compute predicted shares
    s_jt_old        = compute_share_jt(delta_old_jt, xtilde_jt, Gamma)

    # ensure dimensions are correct
    s_jt_old        = reshape(s_jt_old, J, 1)
    s_jt_obs        = reshape(s_jt_obs, J, 1)
    delta_old_jt    = reshape(delta_old_jt, J, 1)
    
    delta_new_jt = delta_old_jt + log.(s_jt_obs) - log.(s_jt_old)

    return delta_new_jt
end

function solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma)

    # Initialize 
    diff_delta   = 100
    delta_old_jt = delta_init
    delta_new_jt = similar(delta_old_jt)

    # Iterate.
    iter_d       = 0
    for its in 1:max_iter_delta

        # update iteration count 
        iter_d += 1
        # update delta
        delta_new_jt = update_delta(delta_old_jt, s_jt_obs, xtilde_jt, Gamma)

        diff_delta = maximum(abs.(delta_old_jt - delta_new_jt))
        delta_old_jt = delta_new_jt

        if diff_delta < tol_delta
            # println("Solved for δ_jt in $iter_d iterations. Difference is $diff_delta")
            break
        end

        # UPDATE
        if iter_d % 200 == 0
            print("Current iteration is $iter_d. Diff = $diff_delta.\n")  
        end
 
    end

    if iter_d == max_iter_delta
        println("Did not solve for delta. Difference is $diff_delta.\n---\n")
    end

    return delta_new_jt

end


# ================================ #
# LOOP OVER MARKETS
# ================================ #
function do_delta_loop(df, Gamma, delta_init)

    delt_mat = zeros(T,J)

    for t in 1:T
        # market index
        # println("---\nNow solving market $t/$T")
        market_idx = (df[:,1] .== t)

        # Get the data
        s_jt_obs = df[market_idx, 3]

        p_jt = df[market_idx, 4]
        x_jt = df[market_idx, 5]
        xtilde_jt = [p_jt x_jt]

        # solve for delta_jt
        delt_mat[t, :] = transpose(solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma))
    end
    delt_vec = vec(transpose(delt_mat))
    return delt_vec
end

function do_GMM_objective(Gamma_vec)
    Gamma_11 = Gamma_vec[1]
    Gamma_21 = Gamma_vec[2]
    Gamma_22 = Gamma_vec[3]
    Gamma = [Gamma_11 0.0; Gamma_21 Gamma_22]
    data = df[:,4:11]
    println("hi1")
    delta_jt  = do_delta_loop(df,Gamma,delta_init)
    println("hi2")
    println(size(delta_jt))
    data[!,"delta_jt"] = delta_jt
    print("hi3")
    # 2sls = 
    model = fit(EconometricModel, # package specific type
            @formula(delta_jt  ~ x + (p ~ z1 + z2 + z3 + z4 + z5 + z6)), # @formula(lhs ~ rhs)
            data # a table
            )
    ξ_jt = residuals(model)
    println("hi4")
    Z = Matrix(df[:,6:11])
    Ω = inv(transpose(Z)*Z)
    objective = (transpose(ξ_jt)*Z)*Ω*(transpose(Z)*ξ_jt) 
    println("hi5")  
    return objective
end

# function add_df_to_GMM(Gamma)
#     objective_df = do_GMM_objective(Gamma,delta_init,df)
#     return objective_df
# end

function do_GMM(Gamma_init)
    Sol = optimize(do_GMM_objective,Gamma_init)
    return Sol
end

Gamma_vec = [100,1.2,1.7]

sol = optimize(do_GMM_objective, Gamma_vec, Optim.Options(show_trace=true, iterations = 100, g_tol = 1e-12), LBFGS())
Optim.minimizer(sol)
# optimize(do_GMM_objective(Gamma),Gamma_init,LBFGS())


# # test function. fix gamma. 
# Gamma_init = [1, 1, 0, 1]
# Gamma = [1 0; 1 1]
# # println("--\nmarket 1....T")
# t = 1
# market_idx = (df[:,1] .== t)

# # Get the data
# s_jt_obs = df[market_idx, 3]

# p_jt = df[market_idx, 4]
# x_jt = df[market_idx, 5]
# xtilde_jt = [p_jt x_jt]
# delta_init = similar(delta_init)

# s_jt_obs[1] = .20

# abc = solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma)

# delt_vec = do_delta_loop(df,Gamma,delta_init)



# do_GMM(Gamma_init)




