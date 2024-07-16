
((ones(1, kFund) - sum(wealthProp_Fund[1:N, 1, :], dims = 1)) .* (wealth_Fund[:, 1] * (1 + r))') + 
(wealth_Fund[:, 1])' .* (sum(wealthProp_Fund[:, 1, :] .* 
(price[:, 1] + dividends[:, 1]) ./ (price[:, 1] * 0.2), dims = 1))

(wealth_Fund[:, 1])'

(sum(wealthProp_Fund[:, 1, :] .* (price[:, 1] + dividends[:, 1]) ./ (price[:, 1] * 0.2), dims = 1))

(wealth_Fund[:, 1])' .* (sum(wealthProp_Fund[:, 1, :] .* (price[:, 1] + dividends[:, 1]) ./ (price[:, 1] * 0.2), dims = 1))

apples = ones(kFund, 5)

apples[:, 1] = ((ones(1, kFund) - sum(wealthProp_Fund[1:N, 1, :], dims = 1)) .* (wealth_Fund[:, 1] * (1 + r))') + 
(wealth_Fund[:, 1])' .* (sum(wealthProp_Fund[:, 1, :] .* 
(price[:, 1] + dividends[:, 1]) ./ (price[:, 1] * 0.2), dims = 1))

apples

price_returns[:, t] = ((price[:, 2] - price[:, 1])/price[:, 1])

pears = ones(N, 5)
pears[:, 2] = pears[:, 2] * 5

pears_p = zeros(N, 5)

pears_p[:, 1] = ((pears[:, 2] - pears[:, 1]) ./ pears[:, 1])
pears

getCovMat(expRet_CovMat_Fund[1:N, 1:N, 2, 1], corr_coef_Fund[1, 1:N], 1, 2)

oranges = expRet_CovMat_Fund[1:N, 1:N, 2, 1]
sqrt(oranges[2, 2, 2, 1])
corr_coef_Fund[1, 1:N]

expRet_CovMat_Fund[1:N, 1:N, 2, :]


###################################################################################################

T = 10
D = dividends
for t in 2:T

    for i in 1:N

        err = rand(Normal(0, 1))                                                # Standard Normal Error Term
        dividends[i, t] = (1 + phi + phi_sd * err) * dividends[i, t-1]          # Expected Dividends for Next Time Period
        fund_val[i, t] = (1 + phi + phi_sd * err) * fund_val[i, t-1]            # Expected Fundamental Value for Next Time Period

        for f in 1:kFund

            expRet_Fund[i, t, f] = (phi * fund_val[i, t-1] + meanR[f] * (fund_val[i, t-1] - price[i, t-1]) + (1 + phi) * dividends[i, t-1] - price[i, t-1]) / price[i, t-1]

            ema_f = exp(-1/ema_wind_Fund[f])
            expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + ((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t-1])^2)

        end

    end
  
end

i = 2
t = 2
f = 5
((phi * fund_val[i, t-1]) + (meanR[f] * (fund_val[i, t-1] - price[i, t-1])) + ((1 + phi) * dividends[i, t-1]) - price[i, t-1]) / price[i, t-1]
ema_f = exp(-1/ema_wind_Fund[f])
(ema_f * 1)
expRet_CovMat_Fund[i, i, t-1, f]

((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t-1])^2)
price_returns[i, t-1]
expRet_Fund[i, t-1, f]

(1/lambda) * inv(expRet_CovMat_Chart[1:N, 1:N, 50, 5])
(expRet_Chart[1:N, 50, 5] .- r)
expRet_Chart[1:N, 50, 5]

((1/lambda) * inv(expRet_CovMat_Chart[1:N, 1:N, 50, 5])) * (expRet_Chart[1:N, 50, 5] .- r)

sum((wealth_Chart[:, 5])' .* wealthProp_Chart[:, 6, :], dims = 2)

# totalPort_Fund[:, 1] = totalPort_Fund[:, 1] + (wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff])
# totalPort_Chart[:, 1] = totalPort_Chart[:, 1] + (wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc])
# price[:, t] = (totalPort_Fund[:, 1] + totalPort_Chart[:, 1]) / assetSupply_max

# Conditional to account for drastic price changes at t == 2
if t == 2

    price[:, 2] = price[:, 1] * 1.1

else

# Determine the price that will Clear each market of Risky Assets
price[:, t] = (totalPort_Fund + totalPort_Chart) / assetSupply_max                

end

for ff in 1:kFund
    wealth = wealth_Fund[ff, t]
    maxWealth = 10 * ()
end

totalPort_Fund = zeros(N, 1)            # Initialise vector of sums of portfolios for Fundamentalists at each time step
totalPort_Chart = zeros(N, 1)           # Initialise vector of sums of portfolios for Chartists at each time step

price[:, 1:1000]

price_returns[:, 100:150]

wealth_Fund[:, 1:10]
wealth_Chart

expRet_Fund[:, 1:300, :]
expRet_Chart[:, 1:300, :]

demand_Fund[:, 270:400, 15]         ### Asset 3 becomes a problem at t = 275
demand_Fund[:, 335:350, 1:3]        ### Asset 3 becomes a problem
demand_Fund
demand_Chart

price[:, 270: 300]
price_returns[:, 270:300]
price_returns

wealthProp_Fund[:, 1:100, 1:3]

s = 30
ttt = 100

round.(sum(demand_Fund[:, ttt, :], dims = 2), digits = 2)
round.(wealth_Fund[:, ttt-1], digits = 2)
round.(wealthProp_Fund[:, ttt, :], digits = 2)
price[:, ttt-1]


potatoes = -((wealth_Chart[:, ttt-1]) .* wealthProp_Chart[iii, ttt, :])/5
all(<(10), potatoes)

(wealth_Chart[:, ttt-1])'
wealthProp_Chart[iii, ttt, :]

#######################################################

if (all(<(asset_price), bound_Fund_min)) && (all(<(asset_price), bound_Chart_min))
                
    continue

elseif (all(<(asset_price), bound_Fund_min)) && !(all(<(asset_price), bound_Chart_min))

    price[ii, t] = maximum(bound_Chart_min)

elseif !(all(<(asset_price), bound_Fund_min)) && (all(<(asset_price), bound_Chart_min))

    price[ii, t] = maximum(bound_Fund_min)
end

if (all(<(asset_price), bound_Fund_max)) && (all(<(asset_price), bound_Chart_max))

    continue

elseif (all(<(asset_price), bound_Fund_max)) && !(all(<(asset_price), bound_Chart_max))

    price[ii, t] = maximum(bound_Chart_max)

elseif !(all(<(asset_price), bound_Fund_max)) && (all(<(asset_price), bound_Chart_max))

    price[ii, t] = maximum(bound_Fund_max)
end

######################################################################

for ii in 1:N

    asset_price = price[ii, t]

    bound_Fund_min = -((wealth_Fund[:, t-1]) .* wealthProp_Fund[ii, t, :])/5
    bound_Chart_min = -((wealth_Chart[:, t-1]) .* wealthProp_Chart[ii, t, :])/5

    bound_Fund_max = ((wealth_Fund[:, t-1]) .* wealthProp_Fund[ii, t, :])/10
    bound_Chart_max = ((wealth_Chart[:, t-1]) .* wealthProp_Chart[ii, t, :])/10
    
    if (all(<(asset_price), bound_Fund_min)) && (all(<(asset_price), bound_Chart_min))
        
        println("The Price is : ", asset_price)
        continue

    elseif (all(<(asset_price), bound_Fund_min)) && !(all(<(asset_price), bound_Chart_min))

        price[ii, t] = price[ii, t-1] + rand(Normal(0, 1))

    elseif !(all(<(asset_price), bound_Fund_min)) && (all(<(asset_price), bound_Chart_min))

        price[ii, t] = price[ii, t-1] + rand(Normal(0, 1))
    end

    if (all(<(asset_price), bound_Fund_max)) && (all(<(asset_price), bound_Chart_max))

        continue

    elseif (all(<(asset_price), bound_Fund_max)) && !(all(<(asset_price), bound_Chart_max))

        price[ii, t] = price[ii, t-1] + rand(Normal(0, 1))

    elseif !(all(<(asset_price), bound_Fund_max)) && (all(<(asset_price), bound_Chart_max))

        price[ii, t] = price[ii, t-1] + rand(Normal(0, 1))
    end
end

############################################################

for f in 1:kFund

    # Fundamentalists Expected Return for the i-th Asset at time t
    expRet_Fund[i, t, f] = ((phi * fund_val[i, t-1]) + 
                            (meanR[f] * (fund_val[i, t-1] - price[i, t-1])) + 
                            ((1 + phi) * dividends[i, t-1]) - 
                            price[i, t-1]) / price[i, t-1]

    # Fundamentalists Exponential Moving Average Parameter
    ema_f = exp(-1/ema_wind_Fund[f])

    # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
    expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + ((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t-1])^2)

end

################################################################

# Ensure Fundamentalists Portfolio does not violate max/min Conditions
for ii in 1:N

    prop = wealthProp_Fund[ii, t, ff]

    if prop > propW_max
        wealthProp_Fund[ii, t, ff] = propW_max
    elseif prop < propW_min
        wealthProp_Fund[ii, t, ff] = propW_min
    else
        continue
    end
end

#####################################################################

sum(wealthProp_Fund[:, 10, :], dims = 1)

# Initialise Max Supply of each Risky Asset
assetSupply_max = (kFund * asset_0 * 100000000) + (kChart * asset_0 * 100000000)       

wealth_Fund[10, 100-1] * wealthProp_Fund[:, 100, 10]

########################################################################

# Sum over the Wealth invested in Risky Assets for all Agents at time t
totalPort_Fund = sum((wealth_Fund[:, t-1])' .* wealthProp_Fund[:, t, :], dims = 2)
totalPort_Chart = sum((wealth_Chart[:, t-1])' .* wealthProp_Chart[:, t, :], dims = 2)

###########################################################################

for ff in 1:kFund

    # Fundamentalists Covariance Matrix of Expected Returns at time t
    expRet_CovMat_Fund[:, :, t, ff] = getCovMat(expRet_CovMat_Fund[:, :, t, ff], corr_coef_Fund[ff, :])

    # Fundamentalists Portfolio of Risky Assets
    wealthProp_Fund[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Fund[:, :, t, ff]) * (expRet_Fund[:, t, ff] .- r)

    # Ensure Fundamentalists Portfolio does not violate max/min Conditions

    for ii in 1:N

        prop = wealthProp_Fund[ii, t, ff]
    
        if prop > propW_max
            wealthProp_Fund[ii, t, ff] = propW_max
        elseif prop < propW_min
            wealthProp_Fund[ii, t, ff] = propW_min
        else
            continue
        end
    end

    # Use Proportional Scaling if conditions violated

    propTot = sum(wealthProp_Fund[:, t, ff], dims = 1)
    propTot = propTot[1]

    if propTot > propW_max
        sf = propW_max ./ propTot
        wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* sf
    elseif propTot < propW_min
        sf = propW_min ./ propTot
        wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* sf
    else
        continue
    end

    wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]
    demand = wealthInvest_Fund[:, t, ff] ./ price[:, t-1]

    for ii in 1:N

        dem = demand[ii]

        if dem > stock_max

            wealthInvest_Fund[ii, t, ff] = price[ii, t-1] * stock_max

        elseif dem < stock_min

            wealthInvest_Fund[ii, t, ff] = price[ii, t-1] * stock_min

        else 
            continue
        end
    end
end

for cc in 1:kChart

    # Chartists Covariance Matrix of Expected Returns at time t
    expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])

    # Chartists Portfolio of Risky Assets
    wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)

    # Ensure Chartists Portfolio does not violate max/min Conditions

    for ii in 1:N

        prop = wealthProp_Chart[ii, t, cc]
    
        if prop > propW_max
            wealthProp_Chart[ii, t, cc] = propW_max
        elseif prop < propW_min
            wealthProp_Chart[ii, t, cc] = propW_min
        else
            continue
        end
    end

    # Use Proportional Scaling if conditions violated
    propTot = sum(wealthProp_Chart[:, t, cc], dims = 1)
    propTot = propTot[1]

    if propTot > propW_max
        sf = propW_max ./ propTot
        wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* sf
    elseif propTot < propW_min
        sf = propW_min ./ propTot
        wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* sf
    else
        continue
    end

    wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]
    demand = wealthInvest_Chart[:, t, cc] ./ price[:, t-1]

    for ii in 1:N

        dem = demand[ii]

        if dem > stock_max

            wealthInvest_Chart[ii, t, cc] = price[ii, t-1] * stock_max

        elseif dem < stock_min

            wealthInvest_Chart[ii, t, cc] = price[ii, t-1] * stock_min

        else 
            continue
        end
    end
end

##############################################################################

wealth_Fund[fff, 2:6]
wealthProp_Fund[:, 2:6, fff]
wealthInvest_Fund[:, 2:6, fff]

##############################################################################

# Conditional to account for drastic price changes at t == 2
if t == 2

    price[:, 2] = price[:, 1] .+ rand(Normal(0, 1))
    println("Fund Port: ", totalPort_Fund)
    println("Chart Port: ", totalPort_Chart)
else

    # Determine the price that will Clear each market of Risky Assets
    price[:, t] = (totalPort_Fund + totalPort_Chart) / assetSupply_max

end

# Condition to ensure that price does not dip below 0

for ii in 1:N
    if price[ii, t] < 0
        price[ii, t] = fund_val[ii, t] + rand(Normal(0, 1))
    end
end

################################################################################

# Update Fundamentalists Wealth at Market Clearing Prices
wealth_Fund[:, t] = ((ones(1, kFund) - sum(wealthProp_Fund[:, t, :], dims = 1)) .* 
(wealth_Fund[:, t-1] * (1 + r))') + 
(wealth_Fund[:, t-1])' .* (sum(wealthProp_Fund[:, t, :] .* 
(price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)

# Update Chartists Wealth at Market Clearing Prices
wealth_Chart[:, t] = ((ones(1, kChart) - sum(wealthProp_Chart[:, t, :], dims = 1)) .* 
 (wealth_Chart[:, t-1] * (1 + r))') + 
 (wealth_Chart[:, t-1])' .* (sum(wealthProp_Chart[:, t, :] .* 
 (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)

###################################################################################