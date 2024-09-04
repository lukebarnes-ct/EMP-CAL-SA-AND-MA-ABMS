##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim

### Parameters

T = 500            # Number of Timesteps
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
propW_min = -0.95   # Min Wealth Investment Proportion 

stock_max = 10      # Max Stock Position
stock_min = -5      # Min Stock Position

### Initialise Variables

cash_0 = 10         # Initial Cash 
div_0 = 0.002       # Initial Dividend
fund_0 = 10         # Initial Fundamental Value
asset_0 = 1         # Initial Risky Asset Positions

assetSupply_max = (kFund * asset_0 * 1) + (kChart * asset_0 * 1)       # Initialise Max Supply of each Risky Asset

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Price Returns of Risky Assets
asset_Returns = zeros(N, T)     # Asset Returns of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

## fill!(expRet_CovMat_Fund, phi)
## fill!(expRet_CovMat_Chart, phi)

expPriceChange_Fund = zeros(N, T, kFund)                 # Fundamentalists Expected Price Change of Risky Assets
expPriceReturn_Chart = zeros(N, T, kChart)                # Chartists Expected Price Return of Risky Assets

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
        expRet_Fund[ii, 1, k] = expPriceChange_Fund[ii, 1, k] + (((1 + phi) * dividends[ii, 1])/price[ii, 1])
        expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k]) + 
        ((1 - ema_f) * (expRet_Fund[ii, 1, k] - 0.0015)^2)
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

        expPriceReturn_Chart[ii, 1, k] = (ema_c * 0.0015)
        expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] + (((1 + phi) * dividends[ii, 1])/price[ii, 1])

        expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k]) + 
        ((1 - ema_c) * (expRet_Chart[ii, 1, k] - 0.0015)^2)
    end

    expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])
end

for t in 2:T

    println("Time is: ", t)

    err = rand(Normal(0, 1), N)                                             # Standard Normal Error Term
    dividends[:, t] = (1 + phi .+ phi_sd * err) .* dividends[:, t-1]        # Expected Dividends for Next Time Period
    fund_val[:, t] = (1 + phi .+ phi_sd * err) .* fund_val[:, t-1]          # Expected Fundamental Value for Next Time Period
    
    for i in 1:N

        for f in 1:kFund

            expPriceChange_Fund[i, t, f] = (phi * fund_val[i, t]) + 
                                           (meanR[f] * (fund_val[i, t] - price[i, t-1]))

            # Fundamentalists Expected Return for the i-th Asset at time t
            expRet_Fund[i, t, f] = (expPriceChange_Fund[i, t, f] + 
                                    ((1 + phi) * dividends[i, t])) / price[i, t-1]
        
            # Fundamentalists Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Fund[f])

            # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + 
                                             ((1 - ema_f) * (expRet_Fund[i, t-1, f] - asset_Returns[i, t])^2)

        end
        
        for c in 1:kChart

            # Chartists Exponential Moving Average Parameter
            ema_c = exp(-1/ema_wind_Chart[c])

            # Conditional to account for price two time periods ago in Expected Price Change
            if t == 2
                expPriceReturn_Chart[i, t, c] = (ema_c * expPriceReturn_Chart[i, t-1, c]) + 
                                                ((1 - ema_c) * (0.01))

            else 
                expPriceReturn_Chart[i, t, c] = (ema_c * expPriceReturn_Chart[i, t-1, c]) + 
                                                ((1 - ema_c) * ((price[i, t-1] - price[i, t-2])/price[i, t-2]))
            end

            # Chartists Expected Return for the i-th Asset at time t
            expRet_Chart[i, t, c] = expPriceReturn_Chart[i, t, c] + (((1 + phi) * dividends[i, t])/price[i, t-1])
            
            # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Chart[i, i, t, c] = (ema_c * expRet_CovMat_Chart[i, i, t-1, c]) + ((1 - ema_c) * (expRet_Chart[i, t-1, c] - asset_Returns[i, t])^2)

        end

    end

    for ff in 1:kFund

        # Fundamentalists Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Fund[:, :, t, ff] = getCovMat(expRet_CovMat_Fund[:, :, t, ff], corr_coef_Fund[ff, :])

        # Fundamentalists Portfolio of Risky Assets
        wealthProp_Fund[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Fund[:, :, t, ff]) * (expRet_Fund[:, t, ff] .- r)

        # Use Proportional Scaling if conditions violated

        propTot = sum(abs.(wealthProp_Fund[:, t, ff]))

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Fund[:, t, ff] = round.(wealthProp_Fund[:, t, ff] .* sf, digits = 3)
        end

        wealthInvest_Fund[:, t, ff] = wealth_Fund[ff, t-1] * wealthProp_Fund[:, t, ff]

    end

    for cc in 1:kChart

        # Chartists Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Chart[:, :, t, cc] = getCovMat(expRet_CovMat_Chart[:, :, t, cc], corr_coef_Chart[cc, :])

        # Chartists Portfolio of Risky Assets
        wealthProp_Chart[:, t, cc] = (1/lambda) * inv(expRet_CovMat_Chart[:, :, t, cc]) * (expRet_Chart[:, t, cc] .- r)

        # Use Proportional Scaling if conditions violated

        propTot = sum(abs.(wealthProp_Chart[:, t, cc]))

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Chart[:, t, cc] = round.(wealthProp_Chart[:, t, cc] .* sf, digits = 3)
        end

        wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]

    end

    identityMat = Matrix{Int}(I, N, N)
    zMat_Fund = wealthProp_Fund[:, t-1, :] ./ price[:, t-1]
    zMat_Chart = wealthProp_Chart[:, t-1, :] ./ price[:, t-1]
    hatMat = identityMat .- (wealthInvest_Fund[:, t, :] * transpose(zMat_Fund)) .- (wealthInvest_Chart[:, t, :] * transpose(zMat_Chart))

    divYield = dividends[:, t] ./ price[:, t-1]
    price[:, t] = inv(hatMat) * ((sum(wealthInvest_Fund[:, t, :] .* 
                  ((wealthProp_Fund[:, t-1, :] .* divYield) .+ 
                   ((1 - sum(wealthProp_Fund[:, t-1, :])) .* (1 + r))), dims = 2)) + 
                   (sum(wealthInvest_Chart[:, t, :] .* 
                   ((wealthProp_Chart[:, t-1, :] .* divYield) .+ 
                    ((1 - sum(wealthProp_Chart[:, t-1, :])) .* (1 + r))), dims = 2))) 

    # Calculate Price Returns
    price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

    # Calculate Asset Returns
    asset_Returns[:, t] = price_returns[:, t] .+ (dividends[:, t] ./ price[:, t-1])

    # Demand for Risky Assets at time t
    demand_Fund[:, t, :] = (wealthInvest_Fund[:, t, :]) ./ price[:, t]
    demand_Chart[:, t, :] = (wealthInvest_Chart[:, t, :]) ./ price[:, t]

    wealthInvest_Fund[:, t, :] = demand_Fund[:, t, :] .* price[:, t]
    wealthInvest_Chart[:, t, :] = demand_Chart[:, t, :] .* price[:, t]

    # Update Fundamentalists Wealth at Market Clearing Prices
    wealth_Fund[:, t] = ((wealth_Fund[:, t-1]' .- sum(wealthInvest_Fund[:, t, :], dims = 1)) .* (1 + r)) + 
                        (sum(wealthInvest_Fund[:, t, :] .* 
                        (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

    wealth_Fund[:, t] = round.(wealth_Fund[:, t], digits = 2)

    # Update Chartists Wealth at Market Clearing Prices
    wealth_Chart[:, t] = ((wealth_Chart[:, t-1]' .- sum(wealthInvest_Chart[:, t, :], dims = 1)) .* (1 + r)) + 
                         (sum(wealthInvest_Chart[:, t, :] .* 
                         (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

    wealth_Chart[:, t] = round.(wealth_Chart[:, t], digits = 2)

end

# Checks (1)

iii = 3
b_ttt = 1
e_ttt = T
fff = 5
ccc = 5

fund_val
dividends

# Checks (2)
expRet_Fund[:, b_ttt:e_ttt, fff]
expRet_Chart[:, b_ttt:e_ttt, ccc]

expPriceReturn_Chart[:, b_ttt:e_ttt, ccc]

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
asset_Returns[:, b_ttt:e_ttt]

# Checks (6)

demand_Fund[:, b_ttt:e_ttt, fff]
demand_Chart[:, b_ttt:e_ttt, ccc]

sum(demand_Fund, dims = 3)
sum(demand_Chart, dims = 3)

sum(demand_Fund, dims = 3) .+ sum(demand_Chart, dims = 3)

# Plot Check

plot(b_ttt:e_ttt, price[iii, b_ttt:e_ttt], label = "Price", title = "Asset i", 
     xlabel = "T", ylabel = "Price", legend = :topright)

plot!(b_ttt:e_ttt, fund_val[iii, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

# Checks (7)

all(wealth_Fund .> 0)
all(wealth_Chart .>= 0)

#####

plot(b_ttt:e_ttt, price[iii, b_ttt:e_ttt], label = "Price", title = "Asset i", 
     xlabel = "T", ylabel = "Price", legend = :topright)

plot!(b_ttt:e_ttt, fund_val[iii, b_ttt:e_ttt], label = "Fundamental Value", linecolor=:red)

