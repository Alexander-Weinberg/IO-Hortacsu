
# ================================ #
# LOAD PACKAGES
# ================================ #
using Random, Statistics, DataFrames, GLM, Distributions, Optim


# ================================ #
# SIMULATE DATA
# ================================ #

Random.seed!(421)

# PARAM
muV                 = 0
sdV                 = 1                 
N                   = 1000              # num auctions 
I                   = 10                # bidders per auction
S                   = 500

# MATRIX OF VALUES
vmat                = muV .+ (sdV .* randn((N, I)))

# ================================ #
# QUESTION 2
# ================================ #
vmat2               = sort(vmat, rev=true, dims=2)                                 # sort within auction by bidders value
avg_v               = mean(vmat2,dims=1)                                           # take the mean of value across auctions
winV                = round(avg_v[1], digits=2)
pay                 = round(avg_v[2], digits=2)

println("The average valuation of the winner = $winV.\n The average payment = $pay")


# ================================ #
# QUESTION 4
# ================================ #

##########################################################################
# Function build_auction_data returns:
# bid_mat: matrix with 1 if there was a bid in period t
# price: matrix that shows the standing price at period t
# bid_submit: matrix that shows that SUBMITTED bids of the bidders (which will be the bidder's value if they bid)
##########################################################################

function build_auction_data(N,I,vmat)
    bid_mat                 = zeros(N,I)
    price                   = missings(Float64,N,I)
    bid_submit              = missings(Float64,N,I)
    
    for nn in 1:N
        global standing_price          = -Inf
        global max_val_so_far          = -Inf
        for ii in 1:I
            vi                  = vmat[nn,ii]

            # Bid
            if vi >= standing_price
                bid_mat[nn,ii]              = 1
                global standing_price       = min(vi,max_val_so_far)
                price[nn,ii]                = standing_price
                bid_submit[nn,ii]           = vi
                global max_val_so_far       = max(vi,max_val_so_far)
            end
        end
    end
    return bid_mat, price, bid_submit
end

bid_mat, price, bid_submit = build_auction_data(N,I,vmat)


N_bidders               = sum(bid_mat, dims=2)
mean(N_bidders)
median(N_bidders)
std(N_bidders)

# ================================ #
# QUESTION 5
# ================================ #

total_bids              = sum(coalesce.(bid_submit,0),dims=2)
avg_bid                 = total_bids ./ (N_bidders)                             
std_avg_bid             = var(avg_bid[(!isnan).(avg_bid)])^.5
mean_avg_bid            = mean(avg_bid[(!isnan).(avg_bid)])
z_stat                  = (N)^(.5)*mean(avg_bid)/std_avg_bid
if abs(z_stat) > 1.96
    println("Reject that the mean bid is equal to zero.")
else 
    println("Fail to reject that the mean bid is equal to zero.")
end                                                                             # t-test rejects

df                      = DataFrame(x=reshape(avg_bid,N))
df                      = df[(!isnan).(df.x),:]
lm(@formula(x ~ 1), df)                                                         # OLS rejects

# ================================ #
# QUESTION 7
# ================================ #

##########################################################################
# Function likelinhood takes inputs:
# y: the k1st highest value of data from a normally distributed iid DGP
# x: the k2st highest value of data from a normally distributed iid DGP
# θ: vector of params [μ,σ] for normal distribution
# Function likelinhood returns:
# lh: value of the likelihood function for the given data and params
##########################################################################

function likelihood(y,x,k1,k2,θ)
    μ                               = θ[1]
    σ                               = θ[2]
    if σ > 0 
        c                               = minimum(x)
        d                               = truncated(Normal(μ,σ), lower = c, upper = Inf)
        first_term                      = log.(factorial(k2 - 1)/(factorial(k2 - k1 - 1)*factorial(k1-1)))
        second_term                     = ((k2 - k1 - 1) .* log.(cdf.(d,y) .- cdf.(d,x))) .+ ((k1-1) .* logccdf.(d,y)) .+ logpdf.(d,y) .- ((k2 - 1) .* logccdf.(d,x))
        log_prob                        = first_term .+ second_term
        lh                              = - mean(log_prob)
    else 
        lh = Inf
    end
    return lh
end


first_period = 1 ## last first period that the first and second highest bidder are allowed to enter
θ = missings(Float64,S,6*first_period)
θ_star = missings(Float64,S,2)
θ12 = ones(first_period,2)
θ13 = ones(first_period,2)
θ23 = ones(first_period,2)
θ_s = ones(first_period,2)
for s = 1:S
    for w = 1:first_period
        θ_init                                  = [2.0 2.0]
        vmat_sim                                = muV .+ (sdV .* randn((N, I)))
        vmat2_sim                               = sort(vmat_sim, rev=true, dims=2)  
        bid_mat_sim, price, bid_submit_sim      = build_auction_data(N,I,vmat_sim)
        bid_submit_sorted                       = sort(coalesce.(bid_submit_sim,-Inf), dims = 2, rev = true) ## sort the bids, replace missings with -Inf so that they go to the end
        index                                   = mapslices(x ->sortperm(x,rev = true), coalesce.(bid_submit_sim,-Inf); dims = 2) # gives the time when the largest, second largest,... bid came in
        idx1                                    = (bid_submit_sorted[:,2] .!= -Inf) .&& (index[:,1] .>= w) # index of auctions where there are atleast 2 highest bids
        idx2                                    = (bid_submit_sorted[:,3] .!= -Inf) .&& (index[:,1] .>= w) # index of auctions where there are atleast 3 highest bids
        y1                                      = bid_submit_sorted[idx1,1]
        y2                                      = bid_submit_sorted[idx2,1]
        y3                                      = bid_submit_sorted[idx2,2]
        x1                                      = bid_submit_sorted[idx1,2]
        x2                                      = bid_submit_sorted[idx2,3]
        x3                                      = bid_submit_sorted[idx2,3]
        res                                     = optimize(θ -> likelihood(vmat2_sim[:,2],vmat2_sim[:,3],2,3,θ), θ_init)
        θ_s[w,:]                                = Optim.minimizer(res)
        res12                                   = optimize(θ -> likelihood(y1,x1,1,2,θ), θ_init)
        θ12[w,:]                                = Optim.minimizer(res12)
        if Optim.converged(res12) == false
            θ12[w,:]                            = [missing missing]
        end
        res13                                   = optimize(θ -> likelihood(y2,x2,1,3,θ), θ_init)
        θ13[w,:]                                = Optim.minimizer(res13)
        if Optim.converged(res13) == false
            θ13[w,:]                            = [missing missing]
        end
        res23                                   = optimize(θ -> likelihood(y3,x3,2,3,θ), θ_init, NelderMead())
        θ23[w,:]                                = Optim.minimizer(res23)
        if Optim.converged(res23) == false
            θ23[w,:]                            = [missing missing]
        end
    end
    θ[s,:]                                      = [vec(θ12'); vec(θ13'); vec(θ23')]'
end

θ_sim_estimates = mean(θ,dims = 1)


