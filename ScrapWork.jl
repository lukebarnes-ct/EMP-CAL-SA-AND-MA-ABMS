
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

#################################################################################

# Update Fundamentalists Wealth at Market Clearing Prices
wealth_Fund[:, t] = ((wealth_Fund[:, 1] .- (sum(wealthInvest_Fund[:, 1, :], dims = 1))') .* (1 + r)) + 
(sum(wealthInvest_Fund[:, t, :] .* 
(price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)

# Update Chartists Wealth at Market Clearing Prices
wealth_Chart[:, t] = (wealth_Chart[:, t-1] .- sum(wealthInvest_Chart[:, t, :], dims = 1) .* (1 + r)) + 
 (sum(wealthInvest_Chart[:, t, :] .* 
 (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)

ddd = ones(3, 20)
sum(ddd)

ggg = [20, 30, 40]

sum(ddd ./ ggg, dims = 2)

if any(x -> x != 0, ggg)
    return 1
else
    return 0
end

########################################################################

TT = 2
xxx = [10.0, 10.0, 10.0]
xxx.^2
optDemand(xxx)

# lower = [0.0, 0.0, 0.0]
res = optimize(optDemand, xxx, NelderMead())
Optim.minimizer(res)
Optim.minimum(res)

yyy = Optim.minimizer(res)
optimize(optDemand, yyy, NelderMead())

priceLowerBounds = [xxx[1] * 0.5, xxx[2] * 0.5, xxx[3] * 0.5]
priceUpperBounds = [xxx[1] * 2, xxx[2] * 2, xxx[3] * 2]

## resOpt = optimize(optDemand, resPrice, NelderMead())
resOpt = optimize(optDemand, priceLowerBounds, priceUpperBounds, 
                  xxx, Fminbox(NelderMead()))

Optim.minimizer(resOpt)
Optim.minimum(resOpt)
                  
########################################################################

# Use Proportional Scaling if conditions violated

propTot = sum(wProp_Fund[:, ff])

if propTot > propW_max
    sf = propW_max ./ propTot
    wProp_Fund[:, ff] = wProp_Fund[:, ff] .* sf

elseif propTot < propW_min
    sf = propW_min ./ propTot
    wProp_Fund[:, ff] = wProp_Fund[:, ff] .* sf
end

# Use Proportional Scaling if conditions violated
propTot = sum(wProp_Chart[:, cc])

if propTot > propW_max
    sf = propW_max ./ propTot
    wProp_Chart[:, cc] = wProp_Chart[:, cc] .* sf

elseif propTot < propW_min
    sf = propW_min ./ propTot
    wProp_Chart[:, cc] = wProp_Chart[:, cc] .* sf
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
end

########################################################################

d_Fund[:, :, 2] = wInvest_Fund ./ assetPrice
d_Chart[:, :, 2] = wInvest_Chart ./ assetPrice

    for iii in 1:N

        for fff in 1:kFund

            prevDem = d_Fund[iii, fff, 1]
            dem = d_Fund[iii, fff, 2]

            if dem > stock_max

                d_Fund[iii, fff, 2] = stock_max

            elseif dem < stock_min

                d_Fund[iii, fff, 2] = stock_min

            end

            if dem < 0
                
                demDiff = prevDem + dem
                
                if demDiff < 0

                    d_Fund[iii, fff, 2] = -prevDem

                end

            end

        end

        for ccc in 1:kChart

            prevDem = d_Chart[iii, ccc, 1]
            dem = d_Chart[iii, ccc, 2]

            if dem > stock_max

                d_Chart[iii, ccc, 2] = stock_max

            elseif dem < stock_min

                d_Chart[iii, ccc, 2] = stock_min

            end

            if dem < 0
                
                demDiff = prevDem + dem
                
                if demDiff < 0

                    d_Chart[iii, ccc, 2] = -prevDem

                end

            end

        end

    end

    totalDemand = sum((d_Fund[:, :, 2]), dims = 2) + 
                  sum((d_Chart[:, :, 2]), dims = 2)

    excessDemand = totalDemand .- assetSupply_max

    # println(excessDemand)

    totalExcessDemand = sum(abs.(excessDemand))

#####################################################################

summation = sum(demand_Fund, dims = 3) .+ sum(demand_Chart, dims = 3)

###################################################################

# Fundamentalists Expected Return for the i-th Asset at time t
expRet_Fund[i, t, f] = (((phi * fund_val[i, t]) + 
(meanR[f] * (fund_val[i, t] - price[i, t])) + 
((1 + phi) * dividends[i, t]) -
price[i, t]) / price[i, t])

# Fundamentalists Expected Return for the i-th Asset at time t
eR_Fund[i, 2, f] = (((phi * fund_val[i, TT]) + 
(meanR[f] * (fund_val[i, TT] - assetPrice[i])) + 
((1 + phi) * dividends[i, TT]) -
assetPrice[i]) / assetPrice[i])

####################################################################

totalDem = sum((d_Fund[:, :, 2]), dims = 2) + 
           sum((d_Chart[:, :, 2]), dims = 2)

for iii in 1:N

    totDem_i = totalDem[iii]
    scaFact = assetSupply_max / totDem_i

    for fff in 1:kFund

        prevDem = d_Fund[iii, fff, 1]
        dem = d_Fund[iii, fff, 2]

        if totDem_i > assetSupply_max

            d_Fund[iii, fff, 2] = dem * scaFact
        end

        dem = d_Fund[iii, fff, 2]

        if dem > stock_max

            d_Fund[iii, fff, 2] = stock_max

        elseif dem < stock_min

            d_Fund[iii, fff, 2] = stock_min

        end

        dem = d_Fund[iii, fff, 2]

        if dem < 0
            
            if prevDem > 0

                demDiff = prevDem + dem
            
                if demDiff < 0

                    d_Fund[iii, fff, 2] = -prevDem

                end

            else 

                d_Fund[iii, fff, 2] = 0
            end

        end

    end

    for ccc in 1:kChart

        prevDem = d_Chart[iii, ccc, 1]
        dem = d_Chart[iii, ccc, 2]

        if totDem_i > assetSupply_max

            d_Chart[iii, ccc, 2] = dem * scaFact
        end

        dem = d_Chart[iii, ccc, 2]

        if dem > stock_max

            d_Chart[iii, ccc, 2] = stock_max

        elseif dem < stock_min

            d_Chart[iii, ccc, 2] = stock_min

        end

        dem = d_Chart[iii, ccc, 2]

        if dem < 0

            if prevDem > 0

                demDiff = prevDem + dem
            
                if demDiff < 0

                    d_Chart[iii, ccc, 2] = -prevDem

                end

            else 

                d_Chart[iii, ccc, 2] = 0
            end

        end

    end

end

totalDem = sum((demand_Fund[:, t, :]), dims = 2) + 
               sum((demand_Chart[:, t, :]), dims = 2)

    for iii in 1:N

        totDem_i = totalDem[iii]
        scaFact = assetSupply_max / totDem_i

        for fff in 1:kFund

            prevDem = demand_Fund[iii, t-1, fff]
            dem = demand_Fund[iii, t, fff]

            if totDem_i > assetSupply_max

                demand_Fund[iii, t, fff] = dem * scaFact
            end

            dem = demand_Fund[iii, t, fff]

            if dem > stock_max

                demand_Fund[iii, t, fff] = stock_max

            elseif dem < stock_min

                demand_Fund[iii, t, fff] = stock_min

            end

            dem = demand_Fund[iii, t, fff]

            if dem < 0
               
                if prevDem > 0

                    demDiff = prevDem + dem
                
                    if demDiff < 0

                        demand_Fund[iii, t, fff] = -prevDem
    
                    end

                else 

                    demand_Fund[iii, t, fff] = 0
                end

            end

        end

        for ccc in 1:kChart

            prevDem = demand_Chart[iii, t-1, ccc]
            dem = demand_Chart[iii, t, ccc]

            if totDem_i > assetSupply_max

                demand_Chart[iii, t, ccc] = dem * scaFact
            end

            dem = demand_Chart[iii, t, ccc]
            
            if dem > stock_max

                demand_Chart[iii, t, ccc] = stock_max

            elseif dem < stock_min

                demand_Chart[iii, t, ccc] = stock_min

            end

            dem = demand_Chart[iii, t, ccc]

            if dem < 0

                if prevDem > 0

                    demDiff = prevDem + dem
                
                    if demDiff < 0

                        demand_Chart[iii, t, ccc] = -prevDem
    
                    end

                else 

                    demand_Chart[iii, t, ccc] = 0
                end

            end

        end

    end

##################################################################

tt = 3
qq = 1

identityMat = Matrix{Float64}(I, N, N)

zMat = wealthProp_Fund[:, tt-1, :] ./ price[:, tt-1]

hatMat = identityMat .- (wealthInvest_Fund[:, tt, :] * transpose(zMat))
invHatMat = inv(hatMat)

divYield = dividends[:, tt] ./ price[:, tt-1]

inv(hatMat) * sum(wealthInvest_Fund[:, tt, :] .* ((wealthProp_Fund[:, tt-1, :] .* divYield) .+ 
                                 ((1 - sum(wealthProp_Fund[:, tt-1, :])) .* (1 + r))), dims = 2)

####################################################################

wealthProp_RF_Fund[t, ff] = (1 - sum(wealthProp_Fund[:, t, ff]))
wealthInvest_RF_Fund[t, ff] = wealth_Fund[ff, t-1] * wealthProp_RF_Fund[t, ff]

wealthProp_RF_Chart[t, cc] = (1 - sum(wealthProp_Chart[:, t, cc]))
wealthInvest_RF_Chart[t, cc] = wealth_Chart[cc, t-1] * wealthProp_RF_Chart[t, cc]

#################################################################################

if dem < 0
                
    demDiff = prevDem + dem
    
    if demDiff < 0

        d_Fund[i, f, 2] = -prevDem

    end

end

if dem < 0
                
    demDiff = prevDem + dem
    
    if demDiff < 0

        d_Chart[i, c, 2] = -prevDem

    end

end

if dem < 0
                
    demDiff = prevDem + dem
    
    if demDiff < 0

        demand_Fund[i, t, f] = -prevDem

    end

end

if dem < 0
                
    demDiff = prevDem + dem
    
    if demDiff < 0

        demand_Chart[i, t, c] = -prevDem

    end

end

##########################################################################

for ff in 1:kFund

    # Fundamentalists Covariance Matrix of Expected Returns at time t
    eR_Cov_Fund[:, :, 2, ff] = getCovMat(eR_Cov_Fund[:, :, 2, ff], corr_coef_Fund[ff, :])

    # Fundamentalists Portfolio of Risky Assets
    wProp_Fund[:, ff] = (1/lambda) * inv(eR_Cov_Fund[:, :, 2, ff]) * 
                        (eR_Fund[:, 2, ff] .- r)

    wProp_Fund[:, ff] = min.(max.(wProp_Fund[:, ff], propW_min), propW_max)

    # Use Proportional Scaling if conditions violated

    propTot = sum(wProp_Fund[:, ff])

    if propTot > propW_max
        sf = propW_max ./ propTot
        wProp_Fund[:, ff] = wProp_Fund[:, ff] .* sf
    end

    wInvest_Fund[:, ff] = wealth_Fund[ff, TT-1] * wProp_Fund[:, ff]

end

for cc in 1:kChart

    # Chartists Covariance Matrix of Expected Returns at time t
    eR_Cov_Chart[:, :, 2, cc] = getCovMat(eR_Cov_Chart[:, :, 2, cc], corr_coef_Chart[cc, :])

    # Chartists Portfolio of Risky Assets
    wProp_Chart[:, cc] = (1/lambda) * inv(eR_Cov_Chart[:, :, 2, cc]) * (eR_Chart[:, 2, cc] .- r)

    wProp_Chart[:, cc] = min.(max.(wProp_Chart[:, cc], propW_min), propW_max)

    # Use Proportional Scaling if conditions violated
    propTot = sum(wProp_Chart[:, cc])

    if propTot > propW_max
        sf = propW_max ./ propTot
        wProp_Chart[:, cc] = wProp_Chart[:, cc] .* sf
    end

    wInvest_Chart[:, cc] = wealth_Chart[cc, TT-1] * wProp_Chart[:, cc]

end

#################################################################################

for i in 1:N

    totDem_i = totalDemand[i]
    scaFact = assetSupply_max / totDem_i

    if (totDem_i > assetSupply_max) | (totDem_i < assetSupply_max)

        d_Fund[i, :, 2] = d_Fund[i, :, 2] .* scaFact
        d_Chart[i, :, 2] = d_Chart[i, :, 2] * scaFact

        totalDemand[i] = sum((d_Fund[i, :, 2])) + sum((d_Chart[i, :, 2]))
    end

end


totalDem = sum((demand_Fund[:, t, :]), dims = 2) + 
sum((demand_Chart[:, t, :]), dims = 2)

for i in 1:N

    totDem_i = totalDem[i]
    scaFact = assetSupply_max / totDem_i

    if (totDem_i > assetSupply_max) | (totDem_i < assetSupply_max)

    demand_Fund[i, t, :] = demand_Fund[i, t, :] .* scaFact
    demand_Chart[i, t, :] = demand_Chart[i, t, :] * scaFact

    end

end

##############################################################################

wProp_Fund[:, ff] = wProp_Fund[:, ff] .* (propW_max / sum(wProp_Fund[:, ff]))
wProp_Chart[:, cc] = wProp_Chart[:, cc] .* (propW_max / sum(wProp_Chart[:, cc]))

wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* (propW_max / sum(wealthProp_Fund[:, t, ff]))
wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* (propW_max / sum(wealthProp_Chart[:, t, cc]))

##############################################################################

println(mse_Array)

perm = sortperm(vec(mse_Array))
ci = CartesianIndices(perm)
ind = ci[perm[1:5]]
t5 = mse_Array[ind]

top5 = sss[1:5]

t5Val = mse_Array[top5]

mse_Array[7]

vec(mse_Array)

##################################################################################

# Update Fundamentalists Investment in the Risk-Free Asset
wealthProp_RF_Fund[:, t] = (1 .- sum(wealthProp_Fund[:, t, :], dims = 1))
wealthInvest_RF_Fund[:, t] = wealth_Fund[:, t-1] .* wealthProp_RF_Fund[:, t]

# Update Chartists Investment in the Risk-Free Asset
wealthProp_RF_Chart[:, t] = (1 .- sum(wealthProp_Chart[:, t, :], dims = 1))
wealthInvest_RF_Chart[:, t] = wealth_Chart[:, t-1] .* wealthProp_RF_Chart[:, t]

for f in 1:kFund

    wp = (1 - sum(wealthProp_Fund[:, t, f]))

    if wp > 1

        wealthProp_RF_Fund[f, t] = 0
        wealthInvest_RF_Fund[f, t] = 0

    else

        # Update Fundamentalists Investment in the Risk-Free Asset
        wealthProp_RF_Fund[f, t] = (1 - sum(wealthProp_Fund[:, t, f]))
        wealthInvest_RF_Fund[f, t] = wealth_Fund[f, t-1] * wealthProp_RF_Fund[f, t]
    end
end

for c in 1:kChart

    wp = (1 - sum(wealthProp_Chart[:, t, c]))

    if wp > 1

        wealthProp_RF_Chart[c, t] = 0
        wealthInvest_RF_Chart[c, t] = 0

    else

        # Update Chartists Investment in the Risk-Free Asset
        wealthProp_RF_Chart[c, t] = (1 - sum(wealthProp_Chart[:, t, c]))
        wealthInvest_RF_Chart[c, t] = wealth_Chart[c, t-1] * wealthProp_RF_Chart[c, t]
    end
end

##################################################################################

### Some Loops

fValues = [4, 8, 20, 40, 100, 200]

for a in fValues

    prices, returns, fundValue, pRet, erFund, erChart, wpFund, wpFund_rf, wpChart, wpChart_rf, 
    wInvFund, wInvFund_rf, wInvChart, wInvChart_rf, wFund, wChart, 
    demFund, demChart, excDem = modelHyperparameters(timeEnd, n, numChart, numFund, 
                                                     wMax, wMin, mRMax, mRMin, 
                                                     corrMax, corrMin, pWMax, pWMin, 
                                                     stockMax, stockMin, a)

    display(plotReturns(returns, BT, ET, numFund, numChart))
    display(plotPrices(prices, fundValue, BT, ET, numFund, numChart))
        
end

##################################################################################

if t > 2
    resPrice = price[:, t-2]
else
    resPrice = price[:, t-1]
end

resOpt = nlsolve(optDemand, resPrice, autodiff = :forward)

##################################################################################

sum((wChart[:, 2] .* wpChart[1, 3, :]) ./ 10)

##################################################################################

wt = length(weeklyData)
jse = plot(1:wt, weeklyData, label = "JSE Top 40", title = "JSE Top 40 Index", 
           xlabel = "T", ylabel = "Index Value", legend = :topleft)

plot(jse, layout = (1, 1), size = (800, 250))

##################################################################################

### Parameters

    # Number of Timesteps
    T = 10000

    # Dividend Mean
    yBar = 1

    # Risk Free Rate
    r = 0.001
    
    # Strength of Fundamentalists Mean Reversion Beliefs
    nu = 1

    # Strength of Trend-Following Chartists Technical Beliefs
    g = 1.89

    sigmaDelta = 0
    sigmaSq = 1
    sigmaEps = 10
    
    # Strength of Memory
    eta = 0

    # Speculators Sensitivity to Mispricing
    alpha = 2000
    
    # Intensity of Choice
    beta = 2

    # Risk Aversion
    lambda = 1/sigmaSq

    # Fundamental Value
    pStar = yBar/r

    ### Initialise Variables and Matrices

    # Dividends of the Risky Asset
    dividends = zeros(T)
    
    # Prices of the Risky Asset
    price = zeros(T)

    # Total Returns of Risky Assets
    returns = zeros(T)

    # Fundamentalists Expected Return of the Risky Asset
    expRet_Fund = zeros(T)

    # Chartists Expected Return of the Risky Asset
    expRet_Chart = zeros(T)

    # Fundamentalists Demand of the Risky Asset
    demand_Fund = zeros(T)

    # Chartists Demand of the Risky Asset
    demand_Chart = zeros(T)

    # Accumulated Profits by Fundamentalists
    accProf_Fund = zeros(T)

    # Accumulated Profits by Chartists
    accProf_Chart = zeros(T)

    # Percentage of Fundamentalists
    n_Fund = zeros(T)

    # Percentage of Chartists
    n_Chart = zeros(T)

    for i in 1:2
        dividends[i] = yBar
        price[i] = 1000
        expRet_Fund[i] = 1000
        expRet_Chart[i] = 1000
    end

    t = 4
        # Delta Error Term
        delta = rand(Normal(0, sigmaDelta), 1)[1]

        dividends[t] = yBar .+ delta

        # Fundamentalists Expected Return at time t+1
        expRet_Fund[t] = pStar + (nu * (price[t-1] - pStar))

        # Chartists Expected Return at time t+1
        expRet_Chart[t] = price[t-1] + (g * (price[t-1] - price[t-2]))

        # Chartists Share of the Risky Asset market at time t
        n_Chart[t] = (1/(1 + exp(beta * (accProf_Fund[t-1] - accProf_Chart[t-1])))) * exp(-((pStar - price[t-1])^2)/alpha)

        # Fundamentalists Share of the Risky Asset market at time t
        n_Fund[t] = 1 - n_Chart[t]

        # Sigma Error Term
        epsilon = rand(Normal(0, sigmaEps), 1)[1]

        # Price of the Risky Asset at time t
        price[t] = (1/(1 + r)) * ((n_Chart[t] * expRet_Chart[t]) + 
                   (n_Fund[t] * expRet_Fund[t]) + 
                    yBar) .+ epsilon

        # Fundamentalists Demand of the Risky Asset at time t
        demand_Fund[t] = (expRet_Fund[t] + dividends[t] - (1 + r) * price[t]) / (lambda * sigmaSq)

        # Chartists Demand of the Risky Asset at time t
        demand_Chart[t] = (expRet_Chart[t] + dividends[t] - (1 + r) * price[t]) / (lambda * sigmaSq)

        # Accumulated Profits by Fundamentalists at time t
        accProf_Fund[t] = (price[t] + dividends[t] - (1 + r) * price[t-1]) * demand_Fund[t-1] + 
        (eta * accProf_Fund[t-1])

        # Accumulated Profits by Chartists at time t
        accProf_Chart = (price[t] + dividends[t] - (1 + r) * price[t-1]) * demand_Chart[t-1] + 
        (eta * accProf_Chart[t-1])

        returns[t] = ((price[t] - price[t-1]) / price[t-1]) + (dividends[t] / price[t-1])

        n1 = n - 1
        p = [mean(price[i:i+n1]) for i in 1:n:length(price) if i+n1 <= length(price)]
        div = [mean(dividends[i:i+n1]) for i in 1:n:length(dividends) if i+n1 <= length(dividends)]
    
        r = zeros(T)
    
        for t in 2:665
    
            r[t] = ((p[t] - p[t-1]) / p[t-1]) + (div[t] / p[t-1])
    
        end

##################################################################################

sf = filter(row -> row.WeeklyClosingPrice >= 52, weeklyPrices)
yyy = 104:116
weeklyPrices[yyy, :]
weekly_JSETOP40_Data[yyy]
filter(row -> row.Week == 1 && row.Year == 2008, dailyPrices)

##################################################################################

t = plotStart:plotEnd_Daily_JSE
plot(t, prices[t], label = "Price", title = "Risky Asset", 
              xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", lw = 1.5, 
              gridlinewidth = 1.5, gridstyle = :dash)

pf =  plot(t, fv[t], label = "Fundamental Value",
    xlabel = "Week", ylabel = "FV", legend = false, framestyle = :box, 
    tick_direction = :none, color = "red", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    plot(1:lengthJSE_Daily, daily_JSETOP40_Data, label = "JSE Top 40", title = "JSE Top 40 Index", 
    xlabel = "Day", ylabel = "Price", legend = false, 
    yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
    tick_direction = :none, color = "purple1", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    weekly_JSETOP40_Data[100:110]

    plot(1:lengthJSE_Weekly, weekly_JSETOP40_Data, label = "JSE Top 40", title = "JSE Top 40 Index", 
    xlabel = "Week", ylabel = "Price", legend = false, 
    yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
    tick_direction = :none, color = "purple1", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    plot(1:lengthSSE50_Daily, daily_SSE50_Data, label = "SSE 50", title = "SSE 50 Index", 
    xlabel = "Day", ylabel = "Price", legend = false, 
    yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
    tick_direction = :none, color = "purple1", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    plot(1:lengthSSE50_Weekly, weekly_SSE50_Data, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "purple1", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)