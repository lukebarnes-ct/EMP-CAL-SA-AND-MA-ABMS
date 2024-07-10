##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions

### Parameters

T = 1000            # Number of Timesteps
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

assetSupply_max = (kFund * asset_0) + (kChart * asset_0)       # Initialise Max Supply of each Risky Asset

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Returns of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

fill!(expRet_CovMat_Fund, phi)
fill!(expRet_CovMat_Chart, phi)

expPriceChange = zeros(N, T, kChart)                # Chartists Expected Price Change of Risky Assets

for i in 1:N
    dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
    fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
    price[i, 1] = fund_0 * 1.5      # Set Initial Asset Price
end

Random.seed!(1234)                  # Set Seed for Reproducibility

# Fundamentalists Mean Reversion Parameter
meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

# Agent's Exponential Moving Average Period
ema_wind_Fund = rand(wind_min:wind_max, kFund)
ema_wind_Chart = rand(wind_min:wind_max, kChart)

# Agent's Expected Correlation Coefficients for the Risky Assets
corr_coef_Fund = round.(rand(Uniform(corr_min, corr_max), kFund, N), digits = 2)
corr_coef_Chart = round.(rand(Uniform(corr_min, corr_max), kChart, N), digits = 2)

wealth_Fund  = zeros(kFund, T)              # Fundamentalists Wealth
wealth_Chart  = zeros(kChart, T)            # Chartists Wealth

wealthProp_Fund = zeros(N, T, kFund)        # Fundamentalists Proportion of Wealth Invested in Risky Assets
wealthProp_Chart = zeros(N, T, kChart)      # Chartists Proportion of Wealth Invested in Risky Assets

demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

for k in 1:kFund
    wealth_Fund[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Fundamentalists

    for ii in 1:N
        wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
        demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 
    end
end

for k in 1:kChart
    wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists

    for ii in 1:N
        wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
        demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 
    end
end

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

for t in 2:T

    println("Time is: ", t)
    for i in 1:N

        err = rand(Normal(0, 1))                                                # Standard Normal Error Term
        dividends[i, t] = (1 + phi + phi_sd * err) * dividends[i, t-1]          # Expected Dividends for Next Time Period
        fund_val[i, t] = (1 + phi + phi_sd * err) * fund_val[i, t-1]            # Expected Fundamental Value for Next Time Period

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
        
        for c in 1:kChart

            # Chartists Exponential Moving Average Parameter
            ema_c = exp(-1/ema_wind_Chart[c])

            # Conditional to account for price two time periods ago in Expected Price Change
            if t == 2
                expPriceChange[i, t, c] = (ema_c * expPriceChange[i, t-1, c]) + ((1 - ema_c) * (0.1))
            
            else 
                expPriceChange[i, t, c] = (ema_c * expPriceChange[i, t-1, c]) + ((1 - ema_c) * ((price[i, t-1] - price[i, t-2])/price[i, t-2]))
            end
            
            # Chartists Expected Return for the i-th Asset at time t
            expRet_Chart[i, t, c] = expPriceChange[i, t, c] + (((1 + phi) * dividends[i, t-1])/price[i, t-1])
            
            # Diagonal of Chartists Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Chart[i, i, t, c] = (ema_c * expRet_CovMat_Chart[i, i, t-1, c]) + ((1 - ema_c) * (expRet_Chart[i, t-1, c] - price_returns[i, t-1])^2)

        end

    end

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
    end

    # Sum over the Wealth invested in Risky Assets for all Agents at time t
    totalPort_Fund = sum((wealth_Fund[:, t-1])' .* wealthProp_Fund[:, t, :], dims = 2)
    totalPort_Chart = sum((wealth_Chart[:, t-1])' .* wealthProp_Chart[:, t, :], dims = 2)

    # Conditional to account for drastic price changes at t == 2
    if t == 2

        price[:, 2] = price[:, 1] * 1.1
        println("Fund Port: ", totalPort_Fund)
        println("Chart Port: ", totalPort_Chart)
    else

        # Determine the price that will Clear each market of Risky Assets
        price[:, t] = (totalPort_Fund + totalPort_Chart) / assetSupply_max

        for ii in 1:N

            asset_price = price[ii, t]

            bound_Fund_min = -((wealth_Fund[:, t-1]) .* wealthProp_Fund[ii, t, :])/5
            bound_Chart_min = -((wealth_Chart[:, t-1]) .* wealthProp_Chart[ii, t, :])/5

            bound_Fund_max = ((wealth_Fund[:, t-1]) .* wealthProp_Fund[ii, t, :])/10
            bound_Chart_max = ((wealth_Chart[:, t-1]) .* wealthProp_Chart[ii, t, :])/10
            
            if (all(<(asset_price), bound_Fund_min)) && (all(<(asset_price), bound_Chart_min))
                
                continue

            elseif (all(<(asset_price), bound_Fund_max)) && (all(<(asset_price), bound_Chart_max))

                continue

            else

                price[ii, t] = 20
            end
        end
    end
    
    # Condition to ensure that price does not dip below 0

    for ii in 1:N
        if price[ii, t] < 0
            price[ii, t] = 0.01
        end
    end
    
    # Calculate Asset Returns
    price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

    # Demand for Risky Assets at time t
    demand_Fund[:, t, :] = ((wealth_Fund[:, t-1])' .* wealthProp_Fund[:, t, :]) ./ price[:, t-1]
    demand_Chart[:, t, :] = ((wealth_Chart[:, t-1])' .* wealthProp_Chart[:, t, :]) ./ price[:, t-1]

    # Update Fundamentalists Wealth at Market Clearing Prices
    wealth_Fund[:, t] = ((ones(1, kFund) - sum(wealthProp_Fund[:, t, :], dims = 1)) .* 
                        (wealth_Fund[:, t-1] * (1 + r))') + 
                        (wealth_Fund[:, t-1])' .* (sum(wealthProp_Fund[:, t, :] .* 
                        (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))

    # Update Chartists Wealth at Market Clearing Prices
    wealth_Chart[:, t] = ((ones(1, kChart) - sum(wealthProp_Chart[:, t, :], dims = 1)) .* 
                         (wealth_Chart[:, t-1] * (1 + r))') + 
                         (wealth_Chart[:, t-1])' .* (sum(wealthProp_Chart[:, t, :] .* 
                         (price[:, t] + dividends[:, t]) ./ (price[:, t-1]), dims = 1))
    

end

# Checks (1)

iii = 3
b_ttt = 1
e_ttt = 100
fff = 10
ccc = 15

fund_val
dividends

# Checks (2)
expRet_Fund[:, b_ttt:e_ttt, fff]
expRet_Chart[:, b_ttt:e_ttt, ccc]

expPriceChange[:, b_ttt:e_ttt, ccc]

# Checks (3)

expRet_CovMat_Fund[:, :, b_ttt:e_ttt, fff]
wealthProp_Fund[:, b_ttt:e_ttt, fff]
wealth_Fund[fff, b_ttt:e_ttt]

# Checks (4)

expRet_CovMat_Chart[:, :, b_ttt:e_ttt, ccc]
wealthProp_Chart[:, b_ttt:e_ttt, ccc]
wealth_Chart[ccc, b_ttt:e_ttt]

# Checks (5)

price[:, b_ttt:e_ttt]
price_returns[:, b_ttt:e_ttt]

# Checks (6)

demand_Fund[:, b_ttt:e_ttt, fff]
demand_Chart[:, b_ttt:e_ttt, ccc]

# Plot Check

plot(b_ttt:e_ttt, price[iii, b_ttt:e_ttt], label = "Asset 1 Price", title = "Asset 1 Price", 
     xlabel = "T", ylabel = "Price", legend = :topright)

