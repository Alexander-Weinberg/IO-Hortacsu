********************************************************************************
* IO II PSet I Q3
* Monica Barbosa & Alex Weinberg
********************************************************************************


cd "/Users/Monica/Documents/UChicago/Classes/Industrial Organization II"
clear all

** Problem 2: Price Elasticity Estimates

import delimited ps1_ex3, clear
describe, d
bysort market: egen total = total(shares)
gen outside = 1 - total

* IV regression to find alpha and beta
gen log_ratio = ln(shares/outside)
ivreg2 log_ratio (price = z) x, gmm2s 
mat b = e(b)
gen alpha = -b[1,1]
gen beta = b[1,2]

* Calculate value of unobserved brand value as the difference between observed 
* shares and utility explained by price and characteristics
gen xi = log_ratio + alpha*prices -beta*x

* Check that xi is calculated correctly
gen share_est = exp(-alpha*prices + beta*x + xi)
bysort market: egen share_est_total = total(share_est)
replace share_est_total = share_est_total + 1
replace share_est = share_est/share_est_total
gen check = shares - share_est
su check, d

* Calculate elasticities using logit elasticity formulas from textbook
forval i = 1/6 {
	bysort market (product): gen prices_`i' = prices[`i']
	bysort market (product): gen shares_`i' = shares[`i']
}

forval i = 1/6 {
	gen pe_`i' = -alpha*prices_`i'*(1-shares_`i')  
	replace pe_`i' = alpha*prices_`i'*shares_`i' if  product != `i'
}

* Take averages across markets and output into matrix
frame copy default pe
frame change pe
collapse  (mean) pe_?, by(product)
mkmat pe* , matrix(pe)
mat list pe
matrix rownames pe = "Product 1" "Product 2" "Product 3" "Product 4" "Product 5" "Product 6"
matrix colnames pe = "Product 1" "Product 2" "Product 3" "Product 4" "Product 5" "Product 6"
 
esttab matrix(pe) using pset1_q3_pe.tex, booktabs nonumbers label nomtitle replace ///
substitute("l*{6}{c}" "l|cccccc")


** Problem 3: Marginal Cost Calculations

* Calculate using FOC for firms
frame change default
gen mc = -1/(alpha*(1-shares)) + prices
frame copy default mc
frame change mc
collapse (mean) mc shares prices, by(product)
list

* Take averages across markets and output into matrix
mkmat mc shares prices, matrix(mc)
mat list mc
matrix rownames mc = "Product 1" "Product 2" "Product 3" "Product 4" "Product 5" "Product 6"
matrix colnames mc = "Marginal Cost" "Share" "Price"
 
esttab matrix(mc) using pset1_q3_mc.tex, booktabs nonumbers label nomtitle replace


** Problem 4: Counterfactual Prices and Shares

frame copy default sim_shares
frame change sim_shares

* Find the price that equates supply and demand in each market
drop if product == 1
gen diff = 100
gen price_old = 1
gen share_top = 1
gen share_bottom = 1
gen price_new = prices
gen share_new = shares
gen share_old = 1
su diff
local diff_max = r(max)
while `diff_max' > .000001 {
	replace share_old = share_new
	replace price_old = price_new
	replace share_top = exp(-alpha*price_old + beta*x + xi)
	drop share_bottom
	bysort market: egen share_bottom = total(share_top)
	replace share_bottom = share_bottom + 1
	replace share_new = share_top/share_bottom
	replace price_new = 1/(alpha*(1-share_new)) + mc
	replace diff = abs(share_old - share_new)
	su diff
	local diff_max = r(max)
}
gen price_counterfactual = price_new

* Calculate counterfactual shares based on counterfactual prices
gen share_counterfactual = exp(-alpha*price_counterfactual + beta*x + xi)
bysort market: egen share_counterfactual_total = total(share_counterfactual)
replace share_counterfactual_total = share_counterfactual_total + 1
replace share_counterfactual = share_counterfactual/share_counterfactual_total

* Data check - new shares should approximately be proportional to old shares
gen share_counterfactual_test = shares + shares*shares_1
gen check_counterfactual = share_counterfactual - share_counterfactual_test
su check_counterfactual, d
gen price_counterfactual_test = 1/(alpha*(1-share_counterfactual)) + mc

* Calculate counterfactual outside share
bysort market: egen total_counterfactual = total(share_counterfactual)
gen outside_counterfactual = 1 - total_counterfactual

preserve
keep market product share_counterfactual price_counterfactual outside_counterfactual
tempfile counterfactual
save `counterfactual'
restore

* Average across markets and output to matrix
collapse (mean) prices shares share_counterfactual share_counterfactual_test price_counterfactual price_counterfactual_test, by(product)

mkmat prices shares price_counterfactual share_counterfactual , matrix(counter)
matrix rownames counter =  "Product 2" "Product 3" "Product 4" "Product 5" "Product 6"
matrix colnames counter = "Price" "Share" "Counterfactual Price" "Counterfactual Share" 
mat list counter
esttab matrix(counter) using pset1_q3_counter.tex, booktabs nonumbers label nomtitle replace

** Problem 5: Counterfactual Profits and Welfare

frame change default 
frame copy default sim_profits
merge 1:1 market product using `counterfactual'
gen profits = shares*(prices - mc)
gen profits_counterfactual = share_counterfactual*(price_counterfactual - mc)
gen welfare = log(1/outside)
replace welfare = . if product != 6
gen welfare_counterfactual = log(1/outside_counterfactual)
replace welfare_counterfactual = . if product ! = 6

gen profit_change = profits_counterfactual - profits
replace profit_change = - profits if product == 1
bysort market: egen total_profit_change = total(profit_change)
gen welfare_change = welfare_counterfactual - welfare

collapse (last) total_profit_change welfare_change, by(market)
su total_profit_change welfare_change




