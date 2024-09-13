##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim

### Parameters

T = 200             # Number of Timesteps
N = 3               # Number of Risky Assets
kChart = 10         # Number of Chartists
kFund = 20          # Number of Fundamentalists

phi = 0.001         # Dividend Growth Rate
phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
r = 0.0012          # Risk Free Rate
lambda = 3          # Relative Risk Aversion

wind_max = 72       # Max Exponential Moving Average Periods
wind_min = 24       # Min Exponential Moving Average Periods

meanR_max = 1        # Max Mean Reversion
meanR_min = 0.75     # Min Mean Reversion

corr_max = 0.5      # Max Expected Correlation Coefficient
corr_min = -0.5     # Min Expected Correlation Coefficient

propW_max = 1    # Max Wealth Investment Proportion
propW_min = 0.05    # Min Wealth Investment Proportion 

stock_max = 10      # Max Stock Position
stock_min = 0       # Min Stock Position

### Initialise Variables

cash_0 = 10         # Initial Cash 
div_0 = 0.002       # Initial Dividend
fund_0 = 5          # Initial Fundamental Value
asset_0 = 1         # Initial Risky Asset Positions

supplyMult = 1      # Asset Supply Multiplier

assetSupply_max = (kFund * asset_0 * supplyMult) + (kChart * asset_0 * supplyMult)       # Initialise Max Supply of each Risky Asset

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Price Returns of Risky Assets
asset_Returns = zeros(N, T)     # Total Returns of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

fill!(expRet_CovMat_Fund, 1)
fill!(expRet_CovMat_Chart, 1)

expPriceChange_Fund = zeros(N, T, kFund)            # Fundamentalists Expected Price Change of Risky Assets
expPriceReturn_Chart = zeros(N, T, kChart)          # Chartists Expected Price Return of Risky Assets

for i in 1:N
    dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
    fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
    price[i, 1] = fund_0 * 0.80     # Set Initial Asset Price
end

Random.seed!(1234)                  # Set Seed for Reproducibility

# Fundamentalists Mean Reversion Parameter
meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

# Agent's Exponential Moving Average Period
ema_wind_Fund = rand(wind_min:wind_max, kFund)
ema_wind_Chart = rand(wind_min:wind_max, kChart)

# Agent's Expected Correlation Coefficients for the Risky Assets
triAg = floor(Int, (N * (N - 1)) / (2))
corr_coef_Fund = round.(rand(Uniform(corr_min, corr_max), 
                        kFund, triAg), digits = 2)

corr_coef_Chart = round.(rand(Uniform(corr_min, corr_max), 
                         kChart, triAg), digits = 2)

wealth_Fund  = zeros(kFund, T)              # Fundamentalists Wealth
wealth_Chart  = zeros(kChart, T)            # Chartists Wealth

wealthProp_Fund = zeros(N, T, kFund)        # Fundamentalists Proportion of Wealth Invested in Risky Assets
wealthProp_Chart = zeros(N, T, kChart)      # Chartists Proportion of Wealth Invested in Risky Assets

wealthProp_RF_Fund = zeros(kFund, T)        # Fundamentalists Proportion of Wealth Invested in Risk-Free Asset
wealthProp_RF_Chart = zeros(kChart, T)      # Chartists Proportion of Wealth Invested in Risk-Free Asset

wealthInvest_Fund = zeros(N, T, kFund)      # Fundamentalists Wealth Invested in Risky Assets
wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

wealthInvest_RF_Fund = zeros(kFund, T)      # Fundamentalists Wealth Invested in Risk-Free Asset
wealthInvest_RF_Chart = zeros(kChart, T)    # Chartists Wealth Invested in Risk-Free Asset

demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

excessDemand_Optim = zeros(1, T)

# Retrieve the Covariance Matrix of Expected Returns for both sets of Agents
function getCovMat(retArr, coefMat)

    index = 1

    for ii in 1:N    

        var_i = sqrt(retArr[ii, ii])

        for ll in 1:N

            var_l = sqrt(retArr[ll, ll])

            if ii == ll
                continue
                
            elseif ll < ii
                continue

            else

                retArr[ii, ll] = coefMat[index] * var_i * var_l
                retArr[ll, ii] = coefMat[index] * var_l * var_i
                index = index + 1
                
            end

        end

    end

    return retArr
end

for k in 1:kFund
    wealthProp_RF_Fund[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
    wealthInvest_RF_Fund[k, 1] = cash_0

    # Fundamentalists Exponential Moving Average Parameter
    ema_f = exp(-1/ema_wind_Fund[k])

    for ii in 1:N
        wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
        wealthInvest_Fund[ii, 1, k] = cash_0

        demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 

        expPriceChange_Fund[ii, 1, k] = (phi * fund_val[ii, 1]) 
        expRet_Fund[ii, 1, k] = expPriceChange_Fund[ii, 1, k] 
        expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k])
    end

    expRet_CovMat_Fund[:, :, 1, k] = getCovMat(expRet_CovMat_Fund[:, :, 1, k], corr_coef_Fund[k, :])

    # Set Initial Wealth of Fundamentalists

    wealth_Fund[k, 1] = wealthInvest_RF_Fund[k, 1] + 
                        sum(wealthInvest_Fund[:, 1, k])   
end

for k in 1:kChart
    wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists
    wealthProp_RF_Chart[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
    wealthInvest_RF_Chart[k, 1] = cash_0

    # Chartists Exponential Moving Average Parameter
    ema_c = exp(-1/ema_wind_Chart[k])

    for ii in 1:N
        wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
        wealthInvest_Chart[ii, 1, k] = cash_0
        
        demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 

        expPriceReturn_Chart[ii, 1, k] = 0
        expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] 

        expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
    end

    expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])

    # Set Initial Wealth of Chartists

    wealth_Chart[k, 1] = wealthInvest_RF_Chart[k, 1] + 
                         sum(wealthInvest_Chart[:, 1, k])          
end

TT = 2

# Find the price such that the excess demand is 0
function optDemand(assetPrice)
    
    eR_Fund = zeros(N, 2, kFund)
    eR_Fund[:, 1, :] = expRet_Fund[:, TT-1, :]
    eR_Cov_Fund = ones(N, N, 2, kFund)
    eR_Cov_Fund[:, :, 1, :] = expRet_CovMat_Fund[:, :, TT-1, :]
    pReturns = (assetPrice .- price[:, TT-1]) ./ price[:, TT-1]
    returns = pReturns .+ (dividends[:, TT] ./ price[:, TT-1])

    eP_Return = zeros(N, 2, kChart)
    eP_Return[:, 1, :] = expPriceReturn_Chart[:, TT-1, :]
    eR_Chart = zeros(N, 2, kChart)
    eR_Chart[:, 1, :] = expRet_Chart[:, TT-1, :]
    eR_Cov_Chart = ones(N, N, 2, kChart)
    eR_Cov_Chart[:, :, 1, :] = expRet_CovMat_Chart[:, :, TT-1, :]

    wProp_Fund = zeros(N, kFund)
    wProp_Chart = zeros(N, kChart)
    wInvest_Fund = zeros(N, kFund)
    wInvest_Chart = zeros(N, kChart)

    d_Fund = zeros(N, kFund, 2)
    d_Fund[:, :, 1] = demand_Fund[:, TT-1, :]
    d_Chart = zeros(N, kChart, 2)
    d_Chart[:, :, 1] = demand_Chart[:, TT-1, :]

    for i in 1:N

        for f in 1:kFund

            ePChange = (phi * fund_val[i, TT]) + 
                       (meanR[f] * (fund_val[i, TT] - assetPrice[i])) 

            # Fundamentalists Expected Return for the i-th Asset at time t
            eR_Fund[i, 2, f] =  (ePChange + 
                                ((1 + phi) * dividends[i, TT])) / assetPrice[i]
        
            # Fundamentalists Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Fund[f])

            # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
            eR_Cov_Fund[i, i, 2, f] = (ema_f * eR_Cov_Fund[i, i, 1, f]) + 
                                      ((1 - ema_f) * (eR_Fund[i, 1, f] - returns[i])^2)

        end

        for c in 1:kChart

            # Chartists Exponential Moving Average Parameter
            ema_c = exp(-1/ema_wind_Chart[c])

            eP_Return[i, 2, c] = (ema_c * eP_Return[i, 1, c]) + 
                                 ((1 - ema_c) * pReturns[i])
            
            # Chartists Expected Return for the i-th Asset at time t
            eR_Chart[i, 2, c] = eP_Return[i, 2, c] + 
                                (((1 + phi) * dividends[i, TT])/assetPrice[i])

            # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
            eR_Cov_Chart[i, i, 2, c] = (ema_c * eR_Cov_Chart[i, i, 1, c]) + 
                                       ((1 - ema_c) * (eR_Chart[i, 1, c] - returns[i])^2)

        end
    end

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

    d_Fund[:, :, 2] = wInvest_Fund ./ assetPrice
    d_Chart[:, :, 2] = wInvest_Chart ./ assetPrice

    for i in 1:N

        for f in 1:kFund

            dem = d_Fund[i, f, 2]

            if dem > stock_max

                d_Fund[i, f, 2] = stock_max

            elseif dem < stock_min

                d_Fund[i, f, 2] = stock_min

            end

        end

        for c in 1:kChart

            dem = d_Chart[i, c, 2]

            if dem > stock_max

                d_Chart[i, c, 2] = stock_max

            elseif dem < stock_min

                d_Chart[i, c, 2] = stock_min

            end

        end

    end

    totalDemand = sum((d_Fund[:, :, 2]), dims = 2) + 
                  sum((d_Chart[:, :, 2]), dims = 2)

    excessDemand = totalDemand .- assetSupply_max

    totalExcessDemand = sum(excessDemand.^2)

    return totalExcessDemand
end

for t in 2:T

    TT = t
    println("Time is: ", TT)

    err = rand(Normal(0, 1), N)                                             # Standard Normal Error Term
    dividends[:, t] = (1 + phi .+ phi_sd * err) .* dividends[:, t-1]        # Expected Dividends for Next Time Period
    fund_val[:, t] = (1 + phi .+ phi_sd * err) .* fund_val[:, t-1]          # Expected Fundamental Value for Next Time Period
    
    resPrice = price[:, t-1]
    resOpt = optimize(optDemand, resPrice, NelderMead())

    # Determine the price that will Clear each market of Risky Assets
    price[:, t] = Optim.minimizer(resOpt)

    excessDemand_Optim[1, t] = round(Optim.minimum(resOpt), digits = 3)

    # Calculate Price Returns
    price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

    # Calculate Asset Returns
    asset_Returns[:, t] = price_returns[:, t] .+ (dividends[:, t] ./ price[:, t-1])

    for i in 1:N

        for f in 1:kFund

            expPriceChange_Fund[i, t, f] = (phi * fund_val[i, t]) + 
                                           (meanR[f] * (fund_val[i, t] - price[i, t])) 

            # Fundamentalists Expected Return for the i-th Asset at time t
            expRet_Fund[i, t, f] = ((expPriceChange_Fund[i, t, f] + 
                                    ((1 + phi) * dividends[i, t])) / price[i, t])
        
            # Fundamentalists Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Fund[f])

            # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + 
                                             ((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t])^2)

        end
        
        for c in 1:kChart

            # Chartists Exponential Moving Average Parameter
            ema_c = exp(-1/ema_wind_Chart[c])

            expPriceReturn_Chart[i, t, c] = (ema_c * expPriceReturn_Chart[i, t-1, c]) + 
                                      ((1 - ema_c) * (price_returns[i, t]))
            
            # Chartists Expected Return for the i-th Asset at time t
            expRet_Chart[i, t, c] = expPriceReturn_Chart[i, t, c] + 
                                    (((1 + phi) * dividends[i, t])/price[i, t])
            
            # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Chart[i, i, t, c] = (ema_c * expRet_CovMat_Chart[i, i, t-1, c]) + 
                                              ((1 - ema_c) * (expRet_Chart[i, t-1, c] - asset_Returns[i, t])^2)

        end

    end

    for ff in 1:kFund

        # Fundamentalists Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Fund[:, :, t, ff] = getCovMat(expRet_CovMat_Fund[:, :, t, ff], corr_coef_Fund[ff, :])

        # Fundamentalists Portfolio of Risky Assets
        wealthProp_Fund[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Fund[:, :, t, ff]) * (expRet_Fund[:, t, ff] .- r)

        # Ensure Fundamentalists Portfolio does not violate max/min Conditions

        wealthProp_Fund[:, t, ff] = min.(max.(wealthProp_Fund[:, t, ff], propW_min), propW_max)

        # Use Proportional Scaling if conditions violated

        propTot = sum(wealthProp_Fund[:, t, ff])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* sf
        end

        wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]

    end

    for cc in 1:kChart

        # Chartists Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])

        # Chartists Portfolio of Risky Assets
        wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)

        # Ensure Chartists Portfolio does not violate max/min Conditions

        wealthProp_Chart[:, t, cc] = min.(max.(wealthProp_Chart[:, t, cc], propW_min), propW_max)

        # Use Proportional Scaling if conditions violated
        propTot = sum(wealthProp_Chart[:, t, cc])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* sf
        end

        wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]

    end

    # Demand for Risky Assets at time t
    demand_Fund[:, t, :] = (wealthInvest_Fund[:, t, :]) ./ price[:, t]
    demand_Chart[:, t, :] = (wealthInvest_Chart[:, t, :]) ./ price[:, t]

    for i in 1:N

        for f in 1:kFund

            dem = demand_Fund[i, t, f]

            if dem > stock_max

                demand_Fund[i, t, f] = stock_max

            elseif dem < stock_min

                demand_Fund[i, t, f] = stock_min

            end

        end

        for c in 1:kChart

            dem = demand_Chart[i, t, c]

            if dem > stock_max

                demand_Chart[i, t, c] = stock_max

            elseif dem < stock_min

                demand_Chart[i, t, c] = stock_min

            end

        end

    end

    for i in 1:N

        for f in 1:kFund

            wealthInvest_Fund[i, t, f] = demand_Fund[i, t, f] * price[i, t]
            wealthProp_Fund[i, t, f] = wealthInvest_Fund[i, t, f] / wealth_Fund[f, t-1]
        end

        for c in 1:kChart

            wealthInvest_Chart[i, t, c] = demand_Chart[i, t, c] * price[i, t]
            wealthProp_Chart[i, t, c] = wealthInvest_Chart[i, t, c] / wealth_Chart[c, t-1]
        end

    end

    # Update Fundamentalists Investment in the Risk-Free Asset
    wealthProp_RF_Fund[:, t] = (1 .- sum(wealthProp_Fund[:, t, :], dims = 1))
    wealthInvest_RF_Fund[:, t] = wealth_Fund[:, t-1] .* wealthProp_RF_Fund[:, t]

    # Update Chartists Investment in the Risk-Free Asset
    wealthProp_RF_Chart[:, t] = (1 .- sum(wealthProp_Chart[:, t, :], dims = 1))
    wealthInvest_RF_Chart[:, t] = wealth_Chart[:, t-1] .* wealthProp_RF_Chart[:, t]

    # Update Fundamentalists Wealth at Market Clearing Prices
    wealth_Fund[:, t] = transpose(wealthInvest_RF_Fund[:, t] .* (1 + r)) + 
                        (sum(wealthInvest_Fund[:, t, :] .* 
                        ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))

    wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)

    # Update Chartists Wealth at Market Clearing Prices
    wealth_Chart[:, t] = transpose(wealthInvest_RF_Chart[:, t] .* (1 + r)) + 
                         (sum(wealthInvest_Chart[:, t, :] .* 
                         ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))

    wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)

end

# Checks (1)

iii = 1
b_ttt = 1
e_ttt = T
fff = 2
ccc = 2

fund_val
dividends

# Checks (2)
expRet_Fund[:, b_ttt:e_ttt, fff]
expRet_CovMat_Fund[:, :, b_ttt:e_ttt, fff]
expRet_Chart[:, b_ttt:e_ttt, ccc]
expRet_CovMat_Chart[:, :, b_ttt:e_ttt, ccc]

# Checks (3)

wealthProp_Fund[:, b_ttt:e_ttt, fff]
wealthProp_RF_Fund[fff, b_ttt:e_ttt]
wealthInvest_Fund[:, b_ttt:e_ttt, fff]
wealthInvest_RF_Fund[fff, b_ttt:e_ttt]
wealth_Fund[fff, b_ttt:e_ttt]

# Checks (4)

wealthProp_Chart[:, b_ttt:e_ttt, ccc]
wealthProp_RF_Chart[ccc, b_ttt:e_ttt]
wealthInvest_Chart[:, b_ttt:e_ttt, ccc]
wealthInvest_RF_Chart[ccc, b_ttt:e_ttt]
wealth_Chart[ccc, b_ttt:e_ttt]

# Checks (5)

price[:, b_ttt:e_ttt]
price_returns[:, b_ttt:e_ttt]
asset_Returns[:, b_ttt:e_ttt]

# Checks (6)

demand_Fund[:, b_ttt:e_ttt, fff]
demand_Chart[:, b_ttt:e_ttt, ccc]

sum(demand_Fund, dims = 3)
sum(demand_Chart, dims = 3)

sum(demand_Fund, dims = 3) .+ sum(demand_Chart, dims = 3)

excessDemand_Optim

# Plot Check Price

p1 = plot(b_ttt:e_ttt, price[1, b_ttt:e_ttt], label = "Price", title = "Asset 1", 
          xlabel = "T", ylabel = "Price", legend = :topleft)

plot!(b_ttt:e_ttt, fund_val[1, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

p2 = plot(b_ttt:e_ttt, price[2, b_ttt:e_ttt], label = "Price", title = "Asset 2", 
          xlabel = "T", ylabel = "Price", legend = :topleft)

plot!(b_ttt:e_ttt, fund_val[2, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

p3 = plot(b_ttt:e_ttt, price[3, b_ttt:e_ttt], label = "Price", title = "Asset 3", 
          xlabel = "T", ylabel = "Price", legend = :topleft)

plot!(b_ttt:e_ttt, fund_val[3, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

plot(p1, p2, p3, layout = (3, 1), size = (800, 800))

# Plot Check Asset Returns

p4 = plot(b_ttt:e_ttt, asset_Returns[1, b_ttt:e_ttt], label = "Returns", title = "Asset 1", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

p5 = plot(b_ttt:e_ttt, asset_Returns[2, b_ttt:e_ttt], label = "Returns", title = "Asset 2", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

p6 = plot(b_ttt:e_ttt, asset_Returns[3, b_ttt:e_ttt], label = "Returns", title = "Asset 3", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

plot(p4, p5, p6, layout = (3, 1), size = (800, 800))

# Histogram Check Asset Returns

h1 = histogram(asset_Returns[1, b_ttt:e_ttt], bins = 80, title = "Histogram of Asset 1 Returns", 
               xlabel = "Prices", ylabel = "Frequency", legend = false)

h2 = histogram(asset_Returns[2, b_ttt:e_ttt], bins = 80, title = "Histogram of Asset 2 Returns", 
               xlabel = "Prices", ylabel = "Frequency", legend = false)

h3 = histogram(asset_Returns[3, b_ttt:e_ttt], bins = 80, title = "Histogram of Asset 3 Returns", 
               xlabel = "Prices", ylabel = "Frequency", legend = false)

plot(h1, h2, h3, layout = (3, 1), size = (800, 800))


# Checks (7)

all(wealth_Fund .>= 0)
all(wealth_Chart .>= 0)

# Checks (8)

sum((price .- fund_val).^2)

##############################################################################

# Optimising Hyperparameters

kF_Seq = 5:5:30
kC_Seq = 5:5:30

wMax_Seq = 24:4:96
wMin_Seq = 4:4:24

rMin_Seq = 0.5:0.05:0.95

cMin_Seq = -0.8:0.1:0.4

mse_Array = zeros(length(kF_Seq), length(kC_Seq), 
                  length(wMax_Seq), length(wMin_Seq),
                  length(rMin_Seq), length(cMin_Seq))

function optHyperparameter(T, N, kF, kC, lam, wMax, wMin, rMin, cMin)

    ### Parameters

    T = T               # Number of Timesteps
    N = N               # Number of Risky Assets
    kChart = kC         # Number of Chartists
    kFund = kF          # Number of Fundamentalists

    phi = 0.001         # Dividend Growth Rate
    phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
    r = 0.0012          # Risk Free Rate
    lambda = lam        # Relative Risk Aversion

    wind_max = wMax     # Max Exponential Moving Average Periods
    wind_min = wMin     # Min Exponential Moving Average Periods

    meanR_max = 1       # Max Mean Reversion
    meanR_min = rMin    # Min Mean Reversion

    corr_max = 0.8      # Max Expected Correlation Coefficient
    corr_min = cMin     # Min Expected Correlation Coefficient

    propW_max = 0.95    # Max Wealth Investment Proportion
    propW_min = 0.00    # Min Wealth Investment Proportion 

    stock_max = 10      # Max Stock Position
    stock_min = 0       # Min Stock Position

    ### Initialise Variables

    cash_0 = 10         # Initial Cash 
    div_0 = 0.002       # Initial Dividend
    fund_0 = 4          # Initial Fundamental Value
    asset_0 = 1         # Initial Risky Asset Positions

    supplyMult = 1      # Asset Supply Multiplier

    assetSupply_max = (kFund * asset_0 * supplyMult) + (kChart * asset_0 * supplyMult)       # Initialise Max Supply of each Risky Asset

    dividends = zeros(N, T)         # Dividends of Risky Assets
    fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
    price = zeros(N, T)             # Prices of Risky Assets
    price_returns = zeros(N, T)     # Price Returns of Risky Assets
    asset_Returns = zeros(N, T)     # Total Returns of Risky Assets

    expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
    expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
    expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
    expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

    fill!(expRet_CovMat_Fund, 1)
    fill!(expRet_CovMat_Chart, 1)

    expPriceChange_Fund = zeros(N, T, kFund)            # Fundamentalists Expected Price Change of Risky Assets
    expPriceReturn_Chart = zeros(N, T, kChart)          # Chartists Expected Price Return of Risky Assets

    for i in 1:N
        dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
        fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
        price[i, 1] = fund_0 * 1.00     # Set Initial Asset Price
    end

    Random.seed!(1234)                  # Set Seed for Reproducibility

    # Fundamentalists Mean Reversion Parameter
    meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

    # Agent's Exponential Moving Average Period
    ema_wind_Fund = rand(wind_min:wind_max, kFund)
    ema_wind_Chart = rand(wind_min:wind_max, kChart)

    # Agent's Expected Correlation Coefficients for the Risky Assets
    triAg = floor(Int, (N * (N - 1)) / (2))
    corr_coef_Fund = round.(rand(Uniform(corr_min, corr_max), 
                            kFund, triAg), digits = 2)

    corr_coef_Chart = round.(rand(Uniform(corr_min, corr_max), 
                            kChart, triAg), digits = 2)

    wealth_Fund  = zeros(kFund, T)              # Fundamentalists Wealth
    wealth_Chart  = zeros(kChart, T)            # Chartists Wealth

    wealthProp_Fund = zeros(N, T, kFund)        # Fundamentalists Proportion of Wealth Invested in Risky Assets
    wealthProp_Chart = zeros(N, T, kChart)      # Chartists Proportion of Wealth Invested in Risky Assets

    wealthProp_RF_Fund = zeros(kFund, T)        # Fundamentalists Proportion of Wealth Invested in Risk-Free Asset
    wealthProp_RF_Chart = zeros(kChart, T)      # Chartists Proportion of Wealth Invested in Risk-Free Asset

    wealthInvest_Fund = zeros(N, T, kFund)      # Fundamentalists Wealth Invested in Risky Assets
    wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

    wealthInvest_RF_Fund = zeros(kFund, T)      # Fundamentalists Wealth Invested in Risk-Free Asset
    wealthInvest_RF_Chart = zeros(kChart, T)    # Chartists Wealth Invested in Risk-Free Asset

    demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
    demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

    excessDemand_Optim = zeros(1, T)

    for k in 1:kFund
        wealthProp_RF_Fund[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
        wealthInvest_RF_Fund[k, 1] = cash_0
    
        # Fundamentalists Exponential Moving Average Parameter
        ema_f = exp(-1/ema_wind_Fund[k])
    
        for ii in 1:N
            wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
            wealthInvest_Fund[ii, 1, k] = cash_0
    
            demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 
    
            expPriceChange_Fund[ii, 1, k] = (phi * fund_val[ii, 1]) 
            expRet_Fund[ii, 1, k] = expPriceChange_Fund[ii, 1, k] 
            expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k])
        end
    
        expRet_CovMat_Fund[:, :, 1, k] = getCovMat(expRet_CovMat_Fund[:, :, 1, k], corr_coef_Fund[k, :])
    
        # Set Initial Wealth of Fundamentalists
    
        wealth_Fund[k, 1] = wealthInvest_RF_Fund[k, 1] + 
                            sum(wealthInvest_Fund[:, 1, k])   
    end
    
    for k in 1:kChart
        wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists
        wealthProp_RF_Chart[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
        wealthInvest_RF_Chart[k, 1] = cash_0
    
        # Chartists Exponential Moving Average Parameter
        ema_c = exp(-1/ema_wind_Chart[k])
    
        for ii in 1:N
            wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
            wealthInvest_Chart[ii, 1, k] = cash_0
            
            demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 
    
            expPriceReturn_Chart[ii, 1, k] = 0 
            expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] 
    
            expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
        end
    
        expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])
    
        # Set Initial Wealth of Chartists
    
        wealth_Chart[k, 1] = wealthInvest_RF_Chart[k, 1] + 
                             sum(wealthInvest_Chart[:, 1, k])          
    end

    for t in 2:T

        TT = t
    
        err = rand(Normal(0, 1), N)                                             # Standard Normal Error Term
        dividends[:, t] = (1 + phi .+ phi_sd * err) .* dividends[:, t-1]        # Expected Dividends for Next Time Period
        fund_val[:, t] = (1 + phi .+ phi_sd * err) .* fund_val[:, t-1]          # Expected Fundamental Value for Next Time Period
        
        resPrice = price[:, t-1]
        resOpt = optimize(optDemand, resPrice, NelderMead())
    
        # Determine the price that will Clear each market of Risky Assets
        price[:, t] = Optim.minimizer(resOpt)
    
        excessDemand_Optim[1, t] = round(Optim.minimum(resOpt), digits = 3)
    
        # Calculate Price Returns
        price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])
    
        # Calculate Asset Returns
        asset_Returns[:, t] = price_returns[:, t] .+ (dividends[:, t] ./ price[:, t-1])
    
        for i in 1:N
    
            for f in 1:kFund
    
                expPriceChange_Fund[i, t, f] = (phi * fund_val[i, t]) + 
                                               (meanR[f] * (fund_val[i, t] - price[i, t])) 
    
                # Fundamentalists Expected Return for the i-th Asset at time t
                expRet_Fund[i, t, f] = ((expPriceChange_Fund[i, t, f] + 
                                        ((1 + phi) * dividends[i, t])) / price[i, t])
            
                # Fundamentalists Exponential Moving Average Parameter
                ema_f = exp(-1/ema_wind_Fund[f])
    
                # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
                expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + 
                                                 ((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t])^2)
    
            end
            
            for c in 1:kChart
    
                # Chartists Exponential Moving Average Parameter
                ema_c = exp(-1/ema_wind_Chart[c])
    
                expPriceReturn_Chart[i, t, c] = (ema_c * expPriceReturn_Chart[i, t-1, c]) + 
                                          ((1 - ema_c) * (price_returns[i, t]))
                
                # Chartists Expected Return for the i-th Asset at time t
                expRet_Chart[i, t, c] = expPriceReturn_Chart[i, t, c] + 
                                        (((1 + phi) * dividends[i, t])/price[i, t])
                
                # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
                expRet_CovMat_Chart[i, i, t, c] = (ema_c * expRet_CovMat_Chart[i, i, t-1, c]) + 
                                                  ((1 - ema_c) * (expRet_Chart[i, t-1, c] - asset_Returns[i, t])^2)
    
            end
    
        end
    
        for ff in 1:kFund
    
            # Fundamentalists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Fund[:, :, t, ff] = getCovMat(expRet_CovMat_Fund[:, :, t, ff], corr_coef_Fund[ff, :])
    
            # Fundamentalists Portfolio of Risky Assets
            wealthProp_Fund[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Fund[:, :, t, ff]) * (expRet_Fund[:, t, ff] .- r)
    
            # Ensure Fundamentalists Portfolio does not violate max/min Conditions
    
            wealthProp_Fund[:, t, ff] = min.(max.(wealthProp_Fund[:, t, ff], propW_min), propW_max)
    
            # Use Proportional Scaling if conditions violated
    
            propTot = sum(wealthProp_Fund[:, t, ff])
    
            if propTot > propW_max
                sf = propW_max ./ propTot
                wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* sf
            end
    
            wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]
    
        end
    
        for cc in 1:kChart
    
            # Chartists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])
    
            # Chartists Portfolio of Risky Assets
            wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)
    
            # Ensure Chartists Portfolio does not violate max/min Conditions
    
            wealthProp_Chart[:, t, cc] = min.(max.(wealthProp_Chart[:, t, cc], propW_min), propW_max)
    
            # Use Proportional Scaling if conditions violated
            propTot = sum(wealthProp_Chart[:, t, cc])
    
            if propTot > propW_max
                sf = propW_max ./ propTot
                wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* sf
            end
    
            wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]
    
        end
    
        # Demand for Risky Assets at time t
        demand_Fund[:, t, :] = (wealthInvest_Fund[:, t, :]) ./ price[:, t]
        demand_Chart[:, t, :] = (wealthInvest_Chart[:, t, :]) ./ price[:, t]
    
        for i in 1:N
    
            for f in 1:kFund
    
                dem = demand_Fund[i, t, f]
    
                if dem > stock_max
    
                    demand_Fund[i, t, f] = stock_max
    
                elseif dem < stock_min
    
                    demand_Fund[i, t, f] = stock_min
    
                end
    
            end
    
            for c in 1:kChart
    
                dem = demand_Chart[i, t, c]
    
                if dem > stock_max
    
                    demand_Chart[i, t, c] = stock_max
    
                elseif dem < stock_min
    
                    demand_Chart[i, t, c] = stock_min
    
                end
    
            end
    
        end
    
        for i in 1:N
    
            for f in 1:kFund
    
                wealthInvest_Fund[i, t, f] = demand_Fund[i, t, f] * price[i, t]
                wealthProp_Fund[i, t, f] = wealthInvest_Fund[i, t, f] / wealth_Fund[f, t-1]
            end
    
            for c in 1:kChart
    
                wealthInvest_Chart[i, t, c] = demand_Chart[i, t, c] * price[i, t]
                wealthProp_Chart[i, t, c] = wealthInvest_Chart[i, t, c] / wealth_Chart[c, t-1]
            end
    
        end
    
        # Update Fundamentalists Investment in the Risk-Free Asset
        wealthProp_RF_Fund[:, t] = (1 .- sum(wealthProp_Fund[:, t, :], dims = 1))
        wealthInvest_RF_Fund[:, t] = wealth_Fund[:, t-1] .* wealthProp_RF_Fund[:, t]
    
        # Update Chartists Investment in the Risk-Free Asset
        wealthProp_RF_Chart[:, t] = (1 .- sum(wealthProp_Chart[:, t, :], dims = 1))
        wealthInvest_RF_Chart[:, t] = wealth_Chart[:, t-1] .* wealthProp_RF_Chart[:, t]
    
        # Update Fundamentalists Wealth at Market Clearing Prices
        wealth_Fund[:, t] = transpose(wealthInvest_RF_Fund[:, t] .* (1 + r)) + 
                            (sum(wealthInvest_Fund[:, t, :] .* 
                            ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))
    
        wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)
    
        # Update Chartists Wealth at Market Clearing Prices
        wealth_Chart[:, t] = transpose(wealthInvest_RF_Chart[:, t] .* (1 + r)) + 
                             (sum(wealthInvest_Chart[:, t, :] .* 
                             ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))
    
        wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)
    
    end

    return price, fund_val

end

for f in 1:length(kF_Seq)

    for c in 1:length(kC_Seq)

        for wMax in 1:length(wMax_Seq)

            for wMin in 1:length(wMin_Seq)

                for rMin in 1:length(rMin_Seq)

                    for cMin in 1:length(cMin_Seq)

                        pr, fV = optHyperparameter(5, 3, kF_Seq[f], kC_Seq[c], 
                                                   3, wMax_Seq[wMax], wMin_Seq[wMin], 
                                                   rMin_Seq[rMin], cMin_Seq[cMin])
                        mse_Array[f, c, wMax, wMin, rMin, cMin] = sum((pr .- fV).^2)            
                        println("Iteration: f: ", kF_Seq[f], " c: ", kC_Seq[c], 
                                " wmax: ", wMax_Seq[wMax], " wmin: ", wMin_Seq[wMin], 
                                " rmin: ", rMin_Seq[rMin], " cmin: ", cMin_Seq[cMin])

                    end
                end
            end
        end
    end
end

# Find the minimum value and its index
min_value, min_index = findmin(mse_Array)
mse_Array[min_index]

min_KF = kF_Seq[min_index[1]]
min_KC = kC_Seq[min_index[2]]

kFund = min_KF
kChart = min_KC

println(mse_Array)

perm = sortperm(vec(mse_Array))
ci = CartesianIndices(perm)
ind = ci[perm[1:5]]
t5 = mse_Array[ind]

top5 = sss[1:5]

t5Val = mse_Array[top5]

mse_Array[7]

vec(mse_Array)

##############################################################################

# Optimising Hyperparameters

kF_Seq = 5:5:30
kC_Seq = 5:5:30

wMax_Seq = 24:4:96
wMin_Seq = 4:4:24

rMin_Seq = 0.5:0.05:0.95

cMin_Seq = -0.8:0.1:0.4

mse_Array = zeros(length(kF_Seq), length(kC_Seq))
prArray = zeros(3, 2, length(kF_Seq), length(kC_Seq))

function optHyperparameter(T, N, kF, kC, wMax, wMin, rMin, cMin)

    ### Parameters

    T = T               # Number of Timesteps
    N = N               # Number of Risky Assets
    kChart = kC         # Number of Chartists
    kFund = kF          # Number of Fundamentalists

    phi = 0.001         # Dividend Growth Rate
    phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
    r = 0.0012          # Risk Free Rate
    lambda = 3          # Relative Risk Aversion

    wind_max = wMax     # Max Exponential Moving Average Periods
    wind_min = wMin     # Min Exponential Moving Average Periods

    meanR_max = 1       # Max Mean Reversion
    meanR_min = rMin    # Min Mean Reversion

    corr_max = 0.8      # Max Expected Correlation Coefficient
    corr_min = cMin     # Min Expected Correlation Coefficient

    propW_max = 0.95    # Max Wealth Investment Proportion
    propW_min = 0.00    # Min Wealth Investment Proportion 

    stock_max = 10      # Max Stock Position
    stock_min = 0       # Min Stock Position

    ### Initialise Variables

    cash_0 = 10         # Initial Cash 
    div_0 = 0.002       # Initial Dividend
    fund_0 = 5          # Initial Fundamental Value
    asset_0 = 1         # Initial Risky Asset Positions

    supplyMult = 1      # Asset Supply Multiplier

    assetSupply_max = (kFund * asset_0 * supplyMult) + (kChart * asset_0 * supplyMult)       # Initialise Max Supply of each Risky Asset

    dividends = zeros(N, T)         # Dividends of Risky Assets
    fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
    price = zeros(N, T)             # Prices of Risky Assets
    price_returns = zeros(N, T)     # Price Returns of Risky Assets
    asset_Returns = zeros(N, T)     # Total Returns of Risky Assets

    expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
    expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
    expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
    expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

    fill!(expRet_CovMat_Fund, 1)
    fill!(expRet_CovMat_Chart, 1)

    expPriceChange_Fund = zeros(N, T, kFund)            # Fundamentalists Expected Price Change of Risky Assets
    expPriceReturn_Chart = zeros(N, T, kChart)          # Chartists Expected Price Return of Risky Assets

    for i in 1:N
        dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
        fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
        price[i, 1] = fund_0 * 1.00     # Set Initial Asset Price
    end

    Random.seed!(1234)                  # Set Seed for Reproducibility

    # Fundamentalists Mean Reversion Parameter
    meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

    # Agent's Exponential Moving Average Period
    ema_wind_Fund = rand(wind_min:wind_max, kFund)
    ema_wind_Chart = rand(wind_min:wind_max, kChart)

    # Agent's Expected Correlation Coefficients for the Risky Assets
    triAg = floor(Int, (N * (N - 1)) / (2))
    corr_coef_Fund = round.(rand(Uniform(corr_min, corr_max), 
                            kFund, triAg), digits = 2)

    corr_coef_Chart = round.(rand(Uniform(corr_min, corr_max), 
                            kChart, triAg), digits = 2)

    wealth_Fund  = zeros(kFund, T)              # Fundamentalists Wealth
    wealth_Chart  = zeros(kChart, T)            # Chartists Wealth

    wealthProp_Fund = zeros(N, T, kFund)        # Fundamentalists Proportion of Wealth Invested in Risky Assets
    wealthProp_Chart = zeros(N, T, kChart)      # Chartists Proportion of Wealth Invested in Risky Assets

    wealthProp_RF_Fund = zeros(kFund, T)        # Fundamentalists Proportion of Wealth Invested in Risk-Free Asset
    wealthProp_RF_Chart = zeros(kChart, T)      # Chartists Proportion of Wealth Invested in Risk-Free Asset

    wealthInvest_Fund = zeros(N, T, kFund)      # Fundamentalists Wealth Invested in Risky Assets
    wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

    wealthInvest_RF_Fund = zeros(kFund, T)      # Fundamentalists Wealth Invested in Risk-Free Asset
    wealthInvest_RF_Chart = zeros(kChart, T)    # Chartists Wealth Invested in Risk-Free Asset

    demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
    demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

    excessDemand_Optim = zeros(1, T)

    for k in 1:kFund
        wealthProp_RF_Fund[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
        wealthInvest_RF_Fund[k, 1] = cash_0

        # Fundamentalists Exponential Moving Average Parameter
        ema_f = exp(-1/ema_wind_Fund[k])

        for ii in 1:N
            wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
            wealthInvest_Fund[ii, 1, k] = cash_0

            demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 

            expPriceChange_Fund[ii, 1, k] = (phi * fund_val[ii, 1]) 
            expRet_Fund[ii, 1, k] = expPriceChange_Fund[ii, 1, k] 
            expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k])
        end

        expRet_CovMat_Fund[:, :, 1, k] = getCovMat(expRet_CovMat_Fund[:, :, 1, k], corr_coef_Fund[k, :])

        # Set Initial Wealth of Fundamentalists

        wealth_Fund[k, 1] = wealthInvest_RF_Fund[k, 1] + 
                            sum(wealthInvest_Fund[:, 1, k])   
    end

    for k in 1:kChart
        wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists
        wealthProp_RF_Chart[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
        wealthInvest_RF_Chart[k, 1] = cash_0

        # Chartists Exponential Moving Average Parameter
        ema_c = exp(-1/ema_wind_Chart[k])

        for ii in 1:N
            wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
            wealthInvest_Chart[ii, 1, k] = cash_0
            
            demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 

            expPriceReturn_Chart[ii, 1, k] = 0
            expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] 

            expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
        end

        expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])

        # Set Initial Wealth of Chartists

        wealth_Chart[k, 1] = wealthInvest_RF_Chart[k, 1] + 
                            sum(wealthInvest_Chart[:, 1, k])          
    end

    for t in 2:T

        TT = t
    
        err = rand(Normal(0, 1), N)                                             # Standard Normal Error Term
        dividends[:, t] = (1 + phi .+ phi_sd * err) .* dividends[:, t-1]        # Expected Dividends for Next Time Period
        fund_val[:, t] = (1 + phi .+ phi_sd * err) .* fund_val[:, t-1]          # Expected Fundamental Value for Next Time Period
        
        resPrice = price[:, t-1]
        resOpt = optimize(optDemand, resPrice, NelderMead())
    
        # Determine the price that will Clear each market of Risky Assets
        price[:, t] = Optim.minimizer(resOpt)
    
        excessDemand_Optim[1, t] = round(Optim.minimum(resOpt), digits = 3)
    
        # Calculate Price Returns
        price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])
    
        # Calculate Asset Returns
        asset_Returns[:, t] = price_returns[:, t] .+ (dividends[:, t] ./ price[:, t-1])
    
        for i in 1:N
    
            for f in 1:kFund
    
                expPriceChange_Fund[i, t, f] = (phi * fund_val[i, t]) + 
                                               (meanR[f] * (fund_val[i, t] - price[i, t])) 
    
                # Fundamentalists Expected Return for the i-th Asset at time t
                expRet_Fund[i, t, f] = ((expPriceChange_Fund[i, t, f] + 
                                        ((1 + phi) * dividends[i, t])) / price[i, t])
            
                # Fundamentalists Exponential Moving Average Parameter
                ema_f = exp(-1/ema_wind_Fund[f])
    
                # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
                expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + 
                                                 ((1 - ema_f) * (expRet_Fund[i, t-1, f] - price_returns[i, t])^2)
    
            end
            
            for c in 1:kChart
    
                # Chartists Exponential Moving Average Parameter
                ema_c = exp(-1/ema_wind_Chart[c])
    
                expPriceReturn_Chart[i, t, c] = (ema_c * expPriceReturn_Chart[i, t-1, c]) + 
                                          ((1 - ema_c) * (price_returns[i, t]))
                
                # Chartists Expected Return for the i-th Asset at time t
                expRet_Chart[i, t, c] = expPriceReturn_Chart[i, t, c] + 
                                        (((1 + phi) * dividends[i, t])/price[i, t])
                
                # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
                expRet_CovMat_Chart[i, i, t, c] = (ema_c * expRet_CovMat_Chart[i, i, t-1, c]) + 
                                                  ((1 - ema_c) * (expRet_Chart[i, t-1, c] - asset_Returns[i, t])^2)
    
            end
    
        end
    
        for ff in 1:kFund
    
            # Fundamentalists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Fund[:, :, t, ff] = getCovMat(expRet_CovMat_Fund[:, :, t, ff], corr_coef_Fund[ff, :])
    
            # Fundamentalists Portfolio of Risky Assets
            wealthProp_Fund[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Fund[:, :, t, ff]) * (expRet_Fund[:, t, ff] .- r)
    
            # Ensure Fundamentalists Portfolio does not violate max/min Conditions
    
            wealthProp_Fund[:, t, ff] = min.(max.(wealthProp_Fund[:, t, ff], propW_min), propW_max)
    
            # Use Proportional Scaling if conditions violated
    
            propTot = sum(wealthProp_Fund[:, t, ff])
    
            if propTot > propW_max
                sf = propW_max ./ propTot
                wealthProp_Fund[:, t, ff] = wealthProp_Fund[:, t, ff] .* sf
            end
    
            wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]
    
        end
    
        for cc in 1:kChart
    
            # Chartists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])
    
            # Chartists Portfolio of Risky Assets
            wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)
    
            # Ensure Chartists Portfolio does not violate max/min Conditions
    
            wealthProp_Chart[:, t, cc] = min.(max.(wealthProp_Chart[:, t, cc], propW_min), propW_max)
    
            # Use Proportional Scaling if conditions violated
            propTot = sum(wealthProp_Chart[:, t, cc])
    
            if propTot > propW_max
                sf = propW_max ./ propTot
                wealthProp_Chart[:, t, cc] = wealthProp_Chart[:, t, cc] .* sf
            end
    
            wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]
    
        end
    
        # Demand for Risky Assets at time t
        demand_Fund[:, t, :] = (wealthInvest_Fund[:, t, :]) ./ price[:, t]
        demand_Chart[:, t, :] = (wealthInvest_Chart[:, t, :]) ./ price[:, t]
    
        for i in 1:N
    
            for f in 1:kFund
    
                dem = demand_Fund[i, t, f]
    
                if dem > stock_max
    
                    demand_Fund[i, t, f] = stock_max
    
                elseif dem < stock_min
    
                    demand_Fund[i, t, f] = stock_min
    
                end
    
            end
    
            for c in 1:kChart
    
                dem = demand_Chart[i, t, c]
    
                if dem > stock_max
    
                    demand_Chart[i, t, c] = stock_max
    
                elseif dem < stock_min
    
                    demand_Chart[i, t, c] = stock_min
    
                end
    
            end
    
        end
    
        for i in 1:N
    
            for f in 1:kFund
    
                wealthInvest_Fund[i, t, f] = demand_Fund[i, t, f] * price[i, t]
                wealthProp_Fund[i, t, f] = wealthInvest_Fund[i, t, f] / wealth_Fund[f, t-1]
            end
    
            for c in 1:kChart
    
                wealthInvest_Chart[i, t, c] = demand_Chart[i, t, c] * price[i, t]
                wealthProp_Chart[i, t, c] = wealthInvest_Chart[i, t, c] / wealth_Chart[c, t-1]
            end
    
        end
    
        # Update Fundamentalists Investment in the Risk-Free Asset
        wealthProp_RF_Fund[:, t] = (1 .- sum(wealthProp_Fund[:, t, :], dims = 1))
        wealthInvest_RF_Fund[:, t] = wealth_Fund[:, t-1] .* wealthProp_RF_Fund[:, t]
    
        # Update Chartists Investment in the Risk-Free Asset
        wealthProp_RF_Chart[:, t] = (1 .- sum(wealthProp_Chart[:, t, :], dims = 1))
        wealthInvest_RF_Chart[:, t] = wealth_Chart[:, t-1] .* wealthProp_RF_Chart[:, t]
    
        # Update Fundamentalists Wealth at Market Clearing Prices
        wealth_Fund[:, t] = transpose(wealthInvest_RF_Fund[:, t] .* (1 + r)) + 
                            (sum(wealthInvest_Fund[:, t, :] .* 
                            ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))
    
        wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)
    
        # Update Chartists Wealth at Market Clearing Prices
        wealth_Chart[:, t] = transpose(wealthInvest_RF_Chart[:, t] .* (1 + r)) + 
                             (sum(wealthInvest_Chart[:, t, :] .* 
                             ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))
    
        wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)
    
    end

    return price, fund_val

end

for f in 1:length(kF_Seq)

    for c in 1:length(kC_Seq)

        pr, fV = optHyperparameter(2, 3, 
                                   kF_Seq[f], kC_Seq[c], 
                                   72, 24, 
                                   0.75, -0.5)
        prArray[:, :, f, c] = pr
        mse_Array[f, c] = sum((pr .- fV).^2)            
        println("Iteration: f: ", kF_Seq[f], " c: ", kC_Seq[c])
        println(pr)

    end
end

# Find the minimum value and its index
min_value, min_index = findmin(mse_Array)
mse_Array[min_index]

min_KF = kF_Seq[min_index[1]]
min_KC = kC_Seq[min_index[2]]

kFund = min_KF
kChart = min_KC

prArray
mse_Array