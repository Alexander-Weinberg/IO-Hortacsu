# cd("/Users/Monica/Documents/UChicago/Classes/Industrial Organization II")
# cd("/Users/Monica/Downloads/IO-Hortacsu-main/pset1")
# println(size(delta_jt),typeof(delta_jt),delta_jt)

# ================================ #
# LOAD PACKAGES
# ================================ #
using CSV, DataFrames, GLM, Optim, Random, LinearAlgebra, Statistics, GLM, Optim

# PARAM
T = 10     # should be 100
J = 6
Ns = 30   # prefer 50

# NUMPARAM
delta_init      = ones(J,1) ./ J
tol_delta       = 1e-8           # prefer 1e-14
max_iter_delta  = 10000

# DRAW SHOCKS
Random.seed!(04211996)
v_mat           = randn(2,Ns)




# ================================ #
# IMPORT DATA
# ================================ #
df      = CSV.read("./input/ps1_ex4.csv", DataFrame)
df      = df[1:J*T,:]

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

    for i in 1:Ns
        # Draw shock i
        v_i = v_mat[:,i]
        
        # Compute individual shares 
        output = compute_share_ijt(δ_jt, xtilde_jt, Gamma, v_i)
        share_ijt[i,:] = transpose(output)
    end

    # compute share
    share_jt = mean(share_ijt, dims=1) 

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

function solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma, verbose=false)

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
            if verbose
                println("Solved for δ_jt in $iter_d iterations. Difference is $diff_delta")
            end
            break
        end

        # UPDATE
        if iter_d % 200 == 0 && verbose
            print("Current iteration is $iter_d. Diff = $diff_delta.\n")  
        end
 
    end

    if iter_d == max_iter_delta
        @warn "Did not solve for delta. Difference is $diff_delta."
    end

    return delta_new_jt

end


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
        x_jt = df[market_idx, 5]
        xtilde_jt = [p_jt x_jt]

        # solve for delta_jt
        delt_mat[t, :] = transpose(solve_delta(delta_init, s_jt_obs, xtilde_jt, Gamma, verbose))
    end
    delt_vec = vec(transpose(delt_mat))
    return delt_vec
end

function compute_residuals(delta_jt, xtilde_jt, Z, Ω_inv)
    # # 2sls using prespec function
    # model = fit(EconometricModel, # package specific type
    #         @formula(delta_jt  ~ x + (p ~ z1 + z2 + z3 + z4 + z5 + z6)), # @formula(lhs ~ rhs)
    #         data # a table
    #         )
    # ξ_jt = residuals(model)

    # Regression 2sls 
    # firststage      = lm(@formula(p ~ x + z1 + z2 + z3 + z4 + z5 + z6), data)
    # p_hat           = predict(firststage)
    # data[!,"phat"]  = p_hat
    # ols             = lm(@formula(delta_jt ~ x + phat), data)
    # ξ_jt            = residuals(ols)
    # beta            = GLM.coef(ols)

    # Manual 2SLS 
    # TODO: MAKE THIS JUST MATRIX
    bread1          = inv(transpose(xtilde_jt) * Z * Ω_inv * transpose(Z) * xtilde_jt) * transpose(xtilde_jt) * Z
    bread2          = transpose(Z) * delta_jt
    β               = bread1 * Ω_inv * bread2
    ξ_jt            = delta_jt - (xtilde_jt * β) 

    # # # check matrix vs. reg 
    # diff_resid      = abs.(ξ_jt - resid_jt)
    # max_diff_r      = maximum(diff_resid)
    # print("Difference is $max_diff_r\n")
    # # # diff_beta       = abs.(ξ_jt - β)
    # # # max_diff        = maximum(diff_resid)
    return ξ_jt, beta
end

function GMM_objective(Gamma_vec)
    println("Now evaluating obj function.")
    # Build Gamma matrix
    Gamma_11    = Gamma_vec[1]
    Gamma_21    = Gamma_vec[2]
    Gamma_22    = Gamma_vec[3]
    Gamma       = [Gamma_11 0.0; Gamma_21 Gamma_22]

    # Get deltas
    delta_jt                = do_delta_loop(df,Gamma,delta_init)

    # Prep data to compute residuals 
    one_jt                  = ones(size(delta_jt))
    Z                       = hcat(one_jt, Matrix(df[:,5:11])) # matrix of instruments includes x1 and ones.
    xtilde_jt               = hcat(one_jt, Matrix(df[:,4:5])) # matrix with ones, p, x1

    # compute GMM error
    Ω_inv                   = inv(transpose(Z)*Z)       # homoskedasticity
    ξ_jt, beta              = compute_residuals(delta_jt, xtilde_jt, Z, Ω_inv)

    objective               = (transpose(ξ_jt)*Z) * Ω_inv * (transpose(Z)*ξ_jt) 

    # robust 
    if isnan(objective)
        objective = 1e10
    end

    return objective, beta
end

# function add_df_to_GMM(Gamma)
#     objective_df = do_GMM_objective(Gamma,delta_init,df)
#     return objective_df
# end

function aux_gmm_obj(Gamma_vec)
    objective, beta   = GMM_objective(Gamma_vec)
    return objective
end


# testing the objective function
Gamma_init          = [1.0,1.0,1.0]  
@profview GMM_objective(Gamma_init)


# =======================================
# TESTING THE OPTIMIZER
test = false
if test 
    Gamma_init          = [1.0,1.0,1.0]  
    Sol                 = optimize(aux_gmm_obj, Gamma_init)
    gamma_answer        = Optim.minimizer(Sol)
    val, beta           = GMM_objective(gamma_answer)

    # print answer
    Gamma_ans           = [gamma_answer[1] 0.0; gamma_answer[2] gamma_answer[3]]
    ans_dict            = Dict("β_0"=>beta[1], 
                                "β_prices"=>beta[2], 
                                "β_characteristics" => beta[3], 
                                "Γ"=>Gamma_ans)
    println("\nSolution = ")
    ans_dict
end


# # Test 1 with reg p on z and then reg d on phat 
# Solution = 
# Dict{String, Any} with 4 entries:
#   "β_0"               => -4.11959
#   "β_characteristics" => 0.0165052
#   "Γ"                 => [5.61021 0.0; 7.11146 …
#   "β_prices"          => 5.40185

# # Test 2 with manual 2sls
# Solution = 
# Dict{String, Any} with 4 entries:
#   "β_0"               => -4.11959
#   "β_characteristics" => 0.0165052
#   "Γ"                 => [5.44464 0.0; 6.72845 …
#   "β_prices"          => 5.40185


# ## TESTING TO FIND PROBLEM WITH Γ
# Gamma_vec = [1.0,1.0,1.0]  
# Gamma_21 = Gamma_vec[2]
# Gamma_22 = Gamma_vec[3]

# # Data
# t                       = 1
# market_idx              = (df[:,1] .== t)
# p_jt                    = df[market_idx, 4]
# x_jt                    = df[market_idx, 5]
# s_jt_obs                = df[market_idx, 3]
# xtilde_jt               = [p_jt x_jt]

# # s_jt_obs[2]             = 0.01

# # STORAGE 
# obj = []
# delta_tester = zeros(J,3) 
# s_tester = zeros(J,3)
# delta_init      = zeros(J,1) ./ J
# delta_init[1]   = 5.0

# for (i, g) in enumerate([1.0, 2.0, 10.0])
#     # unpack gamma
#     Gamma_vec               = [g,1.0,1.0]  
#     # Gamma                   = [Gamma_11 0.0; Gamma_21 Gamma_22]

#     # Test 
#     # dvec, dmat              = do_delta_loop(df, Gamma, delta_init, true)
#     # push!(dvec_list,dvec)

#     # Full function
#     obj_val, dvec            = do_GMM_objective(Gamma_vec)
#     push!(obj,obj_val) 
#     # delta_tester[:,i]       = dvec
# end




# for monica 
# tested and the market_share equations look good. 
# Vary correctly with delta and Gamma (i.e. delta matters less when g big)
# update delta looks weird. s_jt_obs[5] is 36% but new delta spit out is negative?
# do delta loop looks good because converges and deltas vary with gamma
# Objective function spits out a massive number for gamma = 10, small for gamma = 1

# # testing do delta loop
# print("\nγ_11=1.0, \t2.0, \t10.0\n\n")
# diff12 = abs.(dvec_list[1] - dvec_list[2])
# diff13 = abs.(dvec_list[1] - dvec_list[3])
# did    = diff12 - diff13







# sol = optimize(do_GMM_objective, Gamma_vec)
# sol = optimize(do_GMM_objective, Gamma_vec, Optim.Options(show_trace=true, iterations = 100, g_tol = 1e-12), LBFGS())
# Optim.minimizer(sol)
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




