##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim

### Parameters

T = 200             # Number of Timesteps
N = 3               # Number of Risky Assets
kChart = 20         # Number of Chartists
kFund = 20          # Number of Fundamentalists

phi = 0.001         # Dividend Growth Rate
phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
r = 0.0012          # Risk Free Rate
lambda = 3          # Relative Risk Aversion

wind_max = 80       # Max Exponential Moving Average Periods
wind_min = 20       # Min Exponential Moving Average Periods

meanR_max = 1       # Max Mean Reversion
meanR_min = 0.5     # Min Mean Reversion

corr_max = 0.8      # Max Expected Correlation Coefficient
corr_min = -0.2     # Min Expected Correlation Coefficient

propW_max = 0.95    # Max Wealth Investment Proportion
propW_min = 0.00   # Min Wealth Investment Proportion 

stock_max = 10      # Max Stock Position
stock_min = 0      # Min Stock Position

### Initialise Variables

cash_0 = 10         # Initial Cash 
div_0 = 0.002       # Initial Dividend
fund_0 = 10         # Initial Fundamental Value
asset_0 = 1         # Initial Risky Asset Positions

assetSupply_max = (kFund * asset_0 * 1) + (kChart * asset_0 * 1)       # Initialise Max Supply of each Risky Asset

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Returns of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

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

wealthInvest_Fund = zeros(N, T, kFund)      # Fundamentalists Wealth Invested in Risky Assets
wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

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
    wealth_Fund[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Fundamentalists

    # Fundamentalists Exponential Moving Average Parameter
    ema_f = exp(-1/ema_wind_Fund[k])

    for ii in 1:N
        wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
        demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 

        expPriceChange_Fund[ii, 1, k] = (phi * fund_val[ii, 1])
        expRet_Fund[ii, 1, k] = expPriceChange_Fund[ii, 1, k] 
        expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k])
    end

    expRet_CovMat_Fund[:, :, 1, k] = getCovMat(expRet_CovMat_Fund[:, :, 1, k], corr_coef_Fund[k, :])

end

for k in 1:kChart
    wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists

    # Chartists Exponential Moving Average Parameter
    ema_c = exp(-1/ema_wind_Chart[k])

    for ii in 1:N
        wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
        demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 

        expPriceReturn_Chart[ii, 1, k] = (ema_c * 0.015)
        expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] 

        expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
    end

    expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])
end

TT = 2

# Find the price that such that the excess demand is 0
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

        wInvest_Fund[:, ff] = wealth_Fund[ff, TT-1] * wProp_Fund[:, ff]

    end

    for cc in 1:kChart

        # Chartists Covariance Matrix of Expected Returns at time t
        eR_Cov_Chart[:, :, 2, cc] = getCovMat(eR_Cov_Chart[:, :, 2, cc], corr_coef_Chart[cc, :])

        # Chartists Portfolio of Risky Assets
        wProp_Chart[:, cc] = (1/lambda) * inv(eR_Cov_Chart[:, :, 2, cc]) * (eR_Chart[:, 2, cc] .- r)

        wProp_Chart[:, cc] = min.(max.(wProp_Chart[:, cc], propW_min), propW_max)

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

    totalExcessDemand = sum(abs.(excessDemand))

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

        wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]

    end

    for cc in 1:kChart

        # Chartists Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])

        # Chartists Portfolio of Risky Assets
        wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)

        # Ensure Chartists Portfolio does not violate max/min Conditions

        wealthProp_Chart[:, t, cc] = min.(max.(wealthProp_Chart[:, t, cc], propW_min), propW_max)

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

    wealthInvest_Fund[:, t, :] = demand_Fund[:, t, :] .* price[:, t]
    wealthInvest_Chart[:, t, :] = demand_Chart[:, t, :] .* price[:, t]

    # Update Fundamentalists Wealth at Market Clearing Prices
    wealth_Fund[:, t] = ((wealth_Fund[:, t-1]' .- sum(wealthInvest_Fund[:, t, :], dims = 1)) .* (1 + r)) + 
                        (sum(wealthInvest_Fund[:, t, :] .* 
                        ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))

    wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)

    # Update Chartists Wealth at Market Clearing Prices
    wealth_Chart[:, t] = ((wealth_Chart[:, t-1]' .- sum(wealthInvest_Chart[:, t, :], dims = 1)) .* (1 + r)) + 
                         (sum(wealthInvest_Chart[:, t, :] .* 
                         ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))

    wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)

end

# Checks (1)

iii = 2
b_ttt = 1
e_ttt = T
fff = 5
ccc = 5

fund_val
dividends

# Checks (2)
expRet_Fund[:, b_ttt:e_ttt, fff]
expRet_Chart[:, b_ttt:e_ttt, ccc]

# Checks (3)

expRet_CovMat_Fund[:, :, b_ttt:e_ttt, fff]
wealthProp_Fund[:, b_ttt:e_ttt, fff]
wealth_Fund[fff, b_ttt:e_ttt]
wealthInvest_Fund[:, b_ttt:e_ttt, fff]

# Checks (4)

expRet_CovMat_Chart[:, :, b_ttt:e_ttt, ccc]
wealthProp_Chart[:, b_ttt:e_ttt, ccc]
wealth_Chart[ccc, b_ttt:e_ttt]
wealthInvest_Chart[:, b_ttt:e_ttt, ccc]

# Checks (5)

price[:, b_ttt:e_ttt]
price_returns[:, b_ttt:e_ttt]

# Checks (6)

demand_Fund[:, b_ttt:e_ttt, fff]
demand_Chart[:, b_ttt:e_ttt, ccc]

sum(demand_Fund, dims = 3)
sum(demand_Chart, dims = 3)

sum(demand_Fund, dims = 3) .+ sum(demand_Chart, dims = 3)


# Plot Check

p1 = plot(b_ttt:e_ttt, price[1, b_ttt:e_ttt], label = "Price", title = "Asset 1", 
          xlabel = "T", ylabel = "Price", legend = :topright)

plot!(b_ttt:e_ttt, fund_val[1, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

p2 = plot(b_ttt:e_ttt, price[2, b_ttt:e_ttt], label = "Price", title = "Asset 2", 
          xlabel = "T", ylabel = "Price", legend = :topright)

plot!(b_ttt:e_ttt, fund_val[2, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

p3 = plot(b_ttt:e_ttt, price[3, b_ttt:e_ttt], label = "Price", title = "Asset 3", 
          xlabel = "T", ylabel = "Price", legend = :topright)

plot!(b_ttt:e_ttt, fund_val[3, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

plot(p1, p2, p3, layout = (3, 1), size = (800, 800))

# Checks (7)

all(wealth_Fund .> 0)
all(wealth_Chart .>= 0)