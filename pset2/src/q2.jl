#############################
## IO2 PSet 2 Q2
#############################

## CTRL + L clears the workspace

using Random, Distributions, DataFrames, Kronecker, LinearAlgebra, Compat, Statistics, Plots, LaTeXStrings, Optim, LineSearches

Random.seed!(11221997)
T = 100
K = 30
S = 30
G = 25
R = 25 

## Simulation Draws
ϵ_t_mat                                     = randn(K,S)
η_t_mat                                     = randn(S)

###############################
## Create Simulated Data
###############################

## Params
α = 1
β = 2
δ = 6
γ = 3
ρ = 0.8


df = DataFrame(market = 1:T)
df = repeat(df, inner = K)
df[!,"k"]       .= 0
df[!,"x_t"]     .= 0.0
df[!,"z_it"]    .= 0.0
df[!,"ϵ_it"]    .= 0.0
df[!,"η_t"]     .= 0.0
df[!,"ϕ_it"]    .= 0.0
df[!,"n_t"]     .= 0.0
df[!,"v_t"]     .= 0.0
df[!,"n_star"]  .= 0.0

for t in 1:T 
    market_idx                  = (df[:,1] .== t)
    df[market_idx, "k"]         = 1:30
end


for t in 1:T 
    market_idx                  = (df[:,1] .== t)
    df[market_idx, "x_t"]       = exp(randn(1)[1])*ones(K)
    df[market_idx, "z_it"]      = 2^(.5)*randn(K)
    df[market_idx, "ϵ_it"]      = randn(K)
    df[market_idx, "η_t"]       = randn(1)[1]*ones(K)
    df[market_idx, "ϕ_it"]      = df[market_idx, "z_it"].*α + ρ.*df[market_idx, "η_t"] + (1-ρ^2)^(.5).*df[market_idx, "ϵ_it"]
end

## Find equilibrium n
df                              = sort(df, [:market, :ϕ_it], rev = [false,true] )

for t in 1:T
    market_idx                  = (df[:,1]   .== t)
    df[market_idx, "n_t"]       = 1:30
end

df[!,"v_t"]                     = γ*ones(K*T) + df.x_t.*β - δ.*log.(df.n_t) + df.ϕ_it
df[!, "enter"]                  = (df.v_t .+ df.ϕ_it .>= 0)
df                              = transform(groupby(df, :market), :enter => sum => :n_star)
df                              = sort(df, [:market, :k])
df.n_t                          = df.n_star
df                              = select!(df, Not(:n_star))

############################################################
## Function to compute equilibrium n* and I*
# Inputs:
# θ: vector of parameters that we will optimize over
# k: firm index
# x_t: scalar, observed market-level characteristic
# z_t: vector, observed firm-market specific characterisitics
# ϵ_t: vector, unobserved firm-market idiosyncratic shock
# η_t: scalar, unobserved market-level shock
#############################################################

function calculate_equilibrium(θ, k, x_t, z_t, ϵ_t, η_t) 
    α                                   = θ[1]
    β                                   = θ[2]
    δ                                   = θ[3]
    γ                                   = θ[4]
    ρ                                   = θ[5]
    dfsim                               = DataFrame(k = k)
    dfsim[!,"x_t"]                      = x_t
    dfsim[!,"z_t"]                      = z_t
    dfsim[!,"ϵ_t"]                      = ϵ_t
    dfsim[!,"η_t"]                      = η_t
    ϕ_t = ones(length(ϵ_t))
    if abs(ρ) > 1.0
        ϕ_t .= -9999999
    else
        ϕ_t = z_t.*α + ρ.*η_t + (1-ρ^2)^(.5).*ϵ_t
    end
    dfsim[!,"ϕ_t"]                      = ϕ_t
    dfsim                               = sort(dfsim, :ϕ_t, rev = true)
    dfsim[!,"n_t"]                      = 1:30
    dfsim[!,"v_t"]                      = γ*ones(K) + dfsim.x_t.*β - δ.*log.(dfsim.n_t) + dfsim.ϕ_t
    dfsim[!, "enter"]                   = (dfsim.v_t .+ dfsim.ϕ_t .>= 0)
    dfsim                               = transform(dfsim, :enter => sum => :n_star)
    dfsim                               = sort(dfsim, :k)
    n_star                              = dfsim.n_star[1]
    enter_vec                           = dfsim.enter
    return n_star, enter_vec, ϕ_t
end

## test function
# θ = [1,2,6,3,0.8]
# k = df[1:30,"k"]
# x_t = df[1:30,"x_t"]
# z_t = df[1:30,"z_it"]
# ϵ_t = df[1:30,"ϵ_it"]
# η_t = df[1:30,"η_t"]
# calculate_equilibrium(θ, k, x_t, z_t, ϵ_t, η_t) 

############################################################
## Function to simulate expectations of equilibrium n* and I*
# Inputs:
# θ: vector of parameters that we will optimize over
# k: firm index
# x_t: scalar, observed market-level characteristic
# z_t: vector, observed firm-market specific characterisitics
#############################################################

function simulate_equilibrium(θ, k, x_t, z_t, ϵ_t_mat, η_t_mat)
    n_star_vec                                  = ones(S)
    enter_mat                                   = ones(S, K)
    for s in 1:S
        ϵ_t_vec                                 = Vector(ϵ_t_mat[:,s])
        # ϵ_t_vec                                 = Vector(df[1:30,"ϵ_it"])
        η_t_vec                                 = Vector(η_t_mat[s]*ones(K))
        # η_t_vec                                 = Vector(df[1:30,"η_t"])
        n_star_vec[s], enter_mat[s,:]           = calculate_equilibrium(θ, k, x_t, z_t, ϵ_t_vec, η_t_vec)
    end
    n_star                                      = mean(n_star_vec)
    enter_vec                                   = vec(mean(enter_mat, dims = 1))
    return n_star, enter_vec
end

#####################################################################################
## Function to simulate expectations of equilibrium n* and I* across T markets
# Inputs:
# θ: vector of parameters that we will optimize over
# df: data frame with atleast 4 variables:
# - t: indexes markets
# - k: indexes firms
# - x_t: market-level observed characterisitics
# - z_it: firm-market-level observed characteristics
######################################################################################

function loop_simulate_eq(θ, df, ϵ_t_mat, η_t_mat)
    df[!,"n_star"]                      .= 0.0
    df[!, "enter_star"]                 .= 0.0
    for t in 1:T
        market_idx                      = (df[:,1]   .== t)
        n_star, enter_vec               = simulate_equilibrium(θ, df[market_idx,"k"], df[market_idx,"x_t"], df[market_idx,"z_it"], ϵ_t_mat, η_t_mat)
        df[market_idx, "n_star"]        .= n_star
        df[market_idx, "enter_star"]    = enter_vec
    end
    return df
end

######################################################################################
## Functions to simulate expectations of equilibrium n* and I*
# Inputs:
# θ: vector of parameters that we will optimize over
# W: weighting matrix
# df: data frame with atleast 4 variables:
# - t: indexes markets
# - k: indexes firms
# - x_t: market-level observed characterisitics
# - z_it: firm-market-level observed characteristics
#######################################################################################

function compute_GMM_obj(θ, W_inv, df, ϵ_t_mat, η_t_mat)
    df                                          = loop_simulate_eq(θ, df, ϵ_t_mat, η_t_mat)
    u_t                                         = df.n_star - df.n_t
    ξ_it                                        = df.enter_star - df.enter
    x_t                                         = df.x_t
    z_it                                        = df.z_it
    moment1                                     = mean(u_t)
    moment2                                     = mean(ξ_it)
    moment3                                     = mean(x_t.*u_t)
    moment4                                     = mean(z_it.*u_t)
    moment5                                     = mean(x_t.*ξ_it)
    moment6_mat                                 = ones(K*K,T)
    for t in 1:T
        market_idx                              = (df[:,1]   .== t)
        moment6_mat[:,t]                        = z_it[market_idx] ⊗ ξ_it[market_idx]
    end
    moment6                                     = mean(moment6_mat)
    moments                                     = transpose([moment1 moment2 moment3 moment4 moment5 moment6])        
    obj                                         = transpose(moments)*(W_inv\moments)
    return                                      df, obj[1]
end

function aux_compute_GMM_obj(θ, df, W_inv, ϵ_t_mat, η_t_mat)
    df, obj                     = compute_GMM_obj(θ, df, W_inv, ϵ_t_mat, η_t_mat)
    return obj
end

######################################################################################
## Function to minimize GMM objective function
# Inputs:
# θ_init: initial value for vector of parameters that we will optimize over
# W_init: weighting matrix for GMM first step
# df: data frame with atleast 4 variables:
# - t: indexes markets
# - k: indexes firms
# - x_t: market-level observed characterisitics
# - z_it: firm-market-level observed characteristics
#######################################################################################

function do_GMM(θ_init, W_inv_init, df, ϵ_t_mat, η_t_mat)
    # minimize the objective and get estimate of new parameters
    println("First Step GMM")
    res                                         = optimize(θ -> aux_compute_GMM_obj(θ, W_inv_init, df, ϵ_t_mat, η_t_mat), θ_init)
    θ                                           = Optim.minimizer(res)
    # simulate new data with estimate to re-estimate moments
    df                                          = loop_simulate_eq(θ, df, ϵ_t_mat, η_t_mat)
    u_t                                         = df.n_star - df.n_t
    ξ_it                                        = df.enter_star - df.enter
    x_t                                         = df.x_t
    z_it                                        = df.z_it
    moment1                                     = mean(u_t)
    moment2                                     = mean(ξ_it)
    moment3                                     = mean(x_t.*u_t)
    moment4                                     = mean(z_it.*u_t)
    moment5                                     = mean(x_t.*ξ_it)
    moment6_mat                                 = ones(K*K,T)
    for t in 1:T
        market_idx                              = (df[:,1]   .== t)
        moment6_mat[:,t]                        = z_it[market_idx] ⊗ ξ_it[market_idx]
    end
    moment6                                     = mean(moment6_mat)
    moments                                     = transpose([moment1 moment2 moment3 moment4 moment5 moment6])          
    # calculate efficient weight matrix 
    W_inv                                       = moments*transpose(moments)
    println("Second Step GMM with Efficient Weighting Matrix")
    res_efficient                               = optimize(θ_eff -> aux_compute_GMM_obj(θ_eff, W_inv, df, ϵ_t_mat, η_t_mat), θ)
    θ_eff                                       = Optim.minimizer(res_efficient)
    obj                                         = aux_compute_GMM_obj(θ_eff, W_inv, df, ϵ_t_mat, η_t_mat)
    return θ, θ_eff, obj, W_inv
end


#######################################
## GMM Objective Graphs
#######################################

α_grid                              = (-400:400)./100 
β_grid                              = (0:400)./100
δ_grid                              = (200:800)./100
γ_grid                              = (-200:600)./100
ρ_grid                              = (250:950)./1000
W_inv_init                          = I

α_obj = similar(α_grid)
i = 1
for a in α_grid
    θ_test = [a, 2,6,3,0.8]
    df, obj =  compute_GMM_obj(θ_test, W_inv_init, df, ϵ_t_mat, η_t_mat)
    α_obj[i] = obj
    i = i + 1
end

β_obj = similar(β_grid)
i = 1
for b in β_grid
    θ_test = [1,b,6,3,0.8]
    df, obj =  compute_GMM_obj(θ_test, W_inv_init, df, ϵ_t_mat, η_t_mat)
    β_obj[i] = obj
    i = i + 1
end

δ_obj = similar(δ_grid)
i = 1
for d in δ_grid
    θ_test = [1,2,d,3,0.8]
    df, obj =  compute_GMM_obj(θ_test, W_inv_init, df, ϵ_t_mat, η_t_mat)
    δ_obj[i] = obj
    i = i + 1
end

γ_obj = similar(γ_grid)
i = 1
for g in γ_grid
    θ_test = [1,2,6,g,0.8]
    df, obj =  compute_GMM_obj(θ_test, W_inv_init, df, ϵ_t_mat, η_t_mat)
    γ_obj[i] = obj
    i = i + 1
end

ρ_obj = similar(ρ_grid)
i = 1
for r in ρ_grid
    θ_test = [1,2,6,3,r]
    df, obj =  compute_GMM_obj(θ_test, W_inv_init, df, ϵ_t_mat, η_t_mat)
    ρ_obj[i] = obj
    i = i + 1
end

## True θ = [1,2,6,3,0.8]
plot(α_grid, α_obj, label = L"GMM Objective for $\alpha$")
savefig("pset2_gmm_obj_alpha.png")
plot(β_grid, β_obj, label = L"GMM Objective for $\beta$")
savefig("pset2_gmm_obj_beta.png")
plot(δ_grid, δ_obj, label = L"GMM Objective for $\delta$")
savefig("pset2_gmm_obj_delta.png")
plot(γ_grid, γ_obj, label = L"GMM Objective for $\gamma$")
savefig("pset2_gmm_obj_gamma.png")
plot(ρ_grid, ρ_obj, label = L"GMM Objective for $\rho$")
savefig("pset2_gmm_obj_rho.png")

########################################################
## Estimations for Different Initial Conditions
## True θ = [1,2,6,3,0.8]
########################################################

α_init_vec = rand(Uniform(0.0,2.0),R)
β_init_vec = rand(Uniform(0.0,4.0),R)
γ_init_vec = rand(Uniform(3.0,9.0),R)
δ_init_vec = rand(Uniform(0.0,6.0),R)
ρ_init_vec = rand(Uniform(0.7,0.9),R)
θ_init_mat = hcat(α_init_vec, β_init_vec, γ_init_vec, δ_init_vec, ρ_init_vec)
                
θ1                                               = ones(R,5)
θ_eff1                                           = ones(R,5)
obj1                                             = ones(R)
for r in 1:R
    t = time()
    println("Iteration $r, time is $t")
    θ1[r,:], θ_eff1[r,:], obj1[r], W_inv         = do_GMM(θ_init_mat[r,:], W_inv_init, df, ϵ_t_mat, η_t_mat)
end

histogram(θ1[:,1],label = L"Estimates of $\alpha$", bins = -1.25:.5:3.25)
savefig("pset2_alpha_hist_init.png")
histogram(θ1[:,2],label = L"Estimates of $\beta$", bins = -0.25:.5:4.25, xticks = 0:.5:4)
savefig("pset2_beta_hist_init.png")
histogram(θ1[:,3],label = L"Estimates of $\delta$", bins = 1.5:1:10.5, xticks = 2:1:10)
savefig("pset2_delta_hist_init.png")
histogram(θ1[:,4],label = L"Estimates of $\gamma$", bins = -2.5:1:8.5, xticks = -2:1:8)
savefig("pset2_gamma_hist_init.png")
histogram(θ1[:,5],label = L"Estimates of $\rho$", bins = -1.1:.2:1.7, xticks = -1:.2:1.6)
savefig("pset2_rho_hist_init.png")

histogram(θ_eff1[:,1],label = L"Estimates of $\alpha$")
savefig("pset2_alpha_hist_init_eff.png")
histogram(θ_eff1[:,2],label = L"Estimates of $\beta$")
savefig("pset2_beta_hist_init_eff.png")
histogram(θ_eff1[:,3],label = L"Estimates of $\delta$")
savefig("pset2_delta_hist_init_eff.png")
histogram(θ_eff1[:,4],label = L"Estimates of $\gamma$")
savefig("pset2_gamma_hist_init_eff.png")
histogram(θ_eff1[:,5],label = L"Estimates of $\rho$")
savefig("pset2_rho_hist_init_eff.png")

########################################################
## Estimations for Different Simulation Noise
## True θ = [1,2,6,3,0.8]
########################################################

θ_init                                           = [0.75 1.75 5.75 2.75 1.0]
θ2                                               = ones(G,5)
θ_eff2                                           = ones(G,5)
obj2                                             = ones(G)
for g in 1:G
    ϵ_t_mat_new                                  = randn(K,S)
    η_t_mat_new                                  = randn(S)
    t                                            = time()
    println("Iteration $g , time is $t")
    θ2[g,:], θ_eff2[g,:], obj2[g], W_inv         = do_GMM(θ_init, W_inv_init, df, ϵ_t_mat_new, η_t_mat_new)
end

histogram(θ2[:,1],label = L"Estimates of $\alpha$", bins = 0.65:.1:1.55, xticks=0.7:.1:1.5)
savefig("pset2_alpha_hist_noise.png")
histogram(θ2[:,2],label = L"Estimates of $\beta$", bins = 1.55:.1:2.35, xticks = 1.6:.1:2.3)
savefig("pset2_beta_hist_noise.png")
histogram(θ2[:,3],label = L"Estimates of $\delta$", bins = 5.5:.1:6.6, xticks=5.5:.1:6.5)
savefig("pset2_delta_hist_noise.png")
histogram(θ2[:,4],label = L"Estimates of $\gamma$", bins = 1.7:.2:4.1, xticks = 1.8:.2:4.5)
savefig("pset2_gamma_hist_noise.png")
histogram(θ2[:,5],label = L"Estimates of $\rho$", bins = 0.45:.05:1.25, xticks = 0.5:.1:1.2)    
savefig("pset2_rho_hist_noise.png")

histogram(θ_eff2[:,1],label = L"Estimates of $\alpha$")
savefig("pset2_alpha_hist_noise_eff.png")
histogram(θ_eff2[:,2],label = L"Estimates of $\beta$")
savefig("pset2_beta_hist_noise_eff.png")
histogram(θ_eff2[:,3],label = L"Estimates of $\delta$")
savefig("pset2_delta_hist_noise_eff.png")
histogram(θ_eff2[:,4],label = L"Estimates of $\gamma$")
savefig("pset2_gamma_hist_noise_eff.png")
histogram(θ_eff2[:,5],label = L"Estimates of $\rho$")    
savefig("pset2_rho_hist_noise_eff.png")

