
# ================================ #
# LOAD PACKAGES
# ================================ #
using Random, Statistics, DataFrames,GLM


# ================================ #
# SIMULATE DATA
# ================================ #

Random.seed!(421)

# PARAM
muV                 = 0
sdV                 = 1                 
N                   = 1000              # num auctions 
I                   = 10                # bidders per auction

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
# QUESTION 3
# ================================ #
bid_mat                 = zeros(N,I)
bid_submit              = zeros(N,I)

for nn in 1:N
    global standing_price          = 0
    global max_val_so_far          = 0
    for ii in 1:I
        vi                  = vmat[nn,ii]

        # Bid
        if vi >= standing_price
            bid_mat[nn,ii]   = 1
            global standing_price   = min(vi,max_val_so_far)
            bid_submit[nn,ii]= standing_price
            global max_val_so_far   = max(vi,max_val_so_far)
        end
    end
end

N_bidders               = sum(bid_mat, dims=2)
mean(N_bidders)
median(N_bidders)
std(N_bidders)


total_bids              = sum(bid_submit,dims=2)
avg_bid                 = total_bids ./ N_bidders
std_avg_bid             = var(avg_bid[(!isnan).(avg_bid)])
mean_avg_bid            = mean(avg_bid[(!isnan).(avg_bid)])
df                      = DataFrame(x=reshape(avg_bid,N))
df                      = df[(!isnan).(df.x),:]

lm(@formula(x ~ 1), df)

# yes reject .