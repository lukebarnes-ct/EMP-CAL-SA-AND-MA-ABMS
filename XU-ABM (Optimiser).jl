##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim

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

assetSupply_max = (kFund * asset_0 * 1) + (kChart * asset_0 * 1)       # Initialise Max Supply of each Risky Asset

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
    price[i, 1] = fund_0 * 1.05     # Set Initial Asset Price
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

wealthInvest_Fund = zeros(N, T, kFund)      # Fundamentalists Wealth Invested in Risky Assets
wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

demand_Fund = zeros(N, T, kFund)            # Fundamentalists Demand of Risky Assets
demand_Chart = zeros(N, T, kChart)          # Chartists Demand of Risky Assets

for k in 1:kFund
    wealth_Fund[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Fundamentalists

    for ii in 1:N
        wealthProp_Fund[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
        demand_Fund[ii, 1, k] = asset_0             # Set Initial Asset Demand 

        expRet_Fund[ii, 1, k] = (phi * fund_val[ii, 1]) + ((1 + phi) * dividends[ii, 1])
    end
end

for k in 1:kChart
    wealth_Chart[k, 1] = cash_0 * (1 + N)           # Set Initial Wealth of Chartists

    for ii in 1:N
        wealthProp_Chart[ii, 1, k] = 1/(1 + N)      # Set Initial Portfolio Weights
        demand_Chart[ii, 1, k] = asset_0            # Set Initial Asset Demand 

        expRet_Chart[ii, 1, k] = 0.1
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
                expPriceChange[i, t, c] = (ema_c * expPriceChange[i, t-1, c]) + ((1 - ema_c) * (0.01))
            
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
        wealthProp_Fund[:, t, ff] = round.(wealthProp_Fund[:, t, ff], digits = 3)

        # Ensure Fundamentalists Portfolio does not violate max/min Conditions
        # No Short Selling
        wealthProp_Fund[:, t, ff] = max.(wealthProp_Fund[:, t, ff], 0.001)

        # Use Proportional Scaling if conditions violated

        propTot = sum(wealthProp_Fund[:, t, ff], dims = 1)
        propTot = propTot[1]

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
        wealthProp_Chart[:, t, cc] = round.(wealthProp_Chart[:, t, cc], digits = 3)
        
        # Ensure Chartists Portfolio does not violate max/min Conditions
        # No Short Selling
        wealthProp_Chart[:, t, cc] = max.(wealthProp_Chart[:, t, cc], 0.001)

        # Use Proportional Scaling if conditions violated
        propTot = sum(wealthProp_Chart[:, t, cc], dims = 1)
        propTot = propTot[1]

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Chart[:, t, cc] = round.(wealthProp_Chart[:, t, cc] .* sf, digits = 3)
        end

        wealthInvest_Chart[:, t, cc] = wealth_Chart[cc, t-1] * wealthProp_Chart[:, t, cc]



    end

    # Sum over the Wealth invested in Risky Assets for all Agents at time t
    totalPort_Fund = sum(wealthInvest_Fund[:, t, :], dims = 2)
    totalPort_Chart = sum(wealthInvest_Chart[:, t, :], dims = 2)

    # Print Output to determine where issues are arising
    println("Fund Port: ", totalPort_Fund)
    println("Chart Port: ", totalPort_Chart)

    # Determine the price that will Clear each market of Risky Assets
    price[:, t] = (totalPort_Fund + totalPort_Chart) / assetSupply_max
    
    # Calculate Asset Returns
    price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

    # Demand for Risky Assets at time t
    demand_Fund[:, t, :] = ((wealth_Fund[:, t-1])' .* wealthProp_Fund[:, t, :]) ./ price[:, t-1]
    demand_Chart[:, t, :] = ((wealth_Chart[:, t-1])' .* wealthProp_Chart[:, t, :]) ./ price[:, t-1]

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

function optDemand(assetPrice)
    
    eR_Fund = zeros(N, 2, kFund)
    eR_Fund[:, 1, :] = expRet_Fund[:, t-1, :]
    eR_Cov_Fund = ones(N, N, 2, kFund)
    eR_Cov_Fund[:, :, 1, :] = expRet_CovMat_Fund[:, :, t-1, :]
    returns = (assetPrice .- price[:, t-1]) ./ price[:, t-1]

    ePChange = zeros(N, 2, kChart)
    ePChange[:, 1, :] = expPriceChange[:, t-1, :]
    eR_Chart = zeros(N, 2, kChart)
    eR_Chart[:, 1, :] = expRet_Chart[:, t-1, :]
    eR_Cov_Chart = ones(N, N, 2, kChart)
    eR_Cov_Chart[:, :, 1, :] = expRet_CovMat_Chart[:, :, t-1, :]

    wProp_Fund = zeros(N, kFund)
    wProp_Chart = zeros(N, kChart)
    wInvest_Fund = zeros(N, kFund)
    wInvest_Chart = zeros(N, kChart)

    for i in 1:N

        for f in 1:kFund

            # Fundamentalists Expected Return for the i-th Asset at time t
            eR_Fund[i, 2, f] = ((phi * fund_val[i, t]) + 
                                (meanR[f] * (fund_val[i, t] - assetPrice[i])) + 
                                ((1 + phi) * dividends[i, t]) -
                                assetPrice[i]) / assetPrice[i]
        
            # Fundamentalists Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Fund[f])

            # Diagonal of Fundamentalists Covariance Matrix of Expected Returns at time t
            eR_Cov_Fund[i, i, 2, f] = (ema_f * eR_Cov_Fund[i, i, 1, f]) + 
                                      ((1 - ema_f) * (eR_Fund[i, 1, f] - returns[i])^2)

        end

        for c in 1:kChart

            # Chartists Exponential Moving Average Parameter
            ema_c = exp(-1/ema_wind_Chart[c])

            ePChange[i, 2, c] = (ema_c * ePChange[:, 1, :]) + 
                                ((1 - ema_c) * returns[i])
            
            # Chartists Expected Return for the i-th Asset at time t
            eR_Chart[i, 2, c] = ePChange[i, 2, c] + (((1 + phi) * dividends[i, t])/assetPrice[i])
            
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

        # Use Proportional Scaling if conditions violated

        propTot = sum(wProp_Fund[:, ff])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wProp_Fund[:, ff] = wProp_Fund[:, ff] .* sf

        elseif propTot < propW_max
            sf = propW_min ./ propTot
            wProp_Fund[:, ff] = wProp_Fund[:, ff] .* sf
        end

        wInvest_Fund[:, ff] = wealth_Fund[ff, t-1] * wProp_Fund[:, ff]

    end

    for cc in 1:kChart

        # Chartists Covariance Matrix of Expected Returns at time t
        eR_Cov_Chart[:, :, 2, cc] = getCovMat(eR_Cov_Chart[:, :, 2, cc], corr_coef_Chart[cc, :])

        # Chartists Portfolio of Risky Assets
        wProp_Chart[:, cc] = (1/lambda) * inv(eR_Cov_Chart[:, :, 2, cc]) * (eR_Chart[:, 2, cc] .- r)

        # Use Proportional Scaling if conditions violated
        propTot = sum(wProp_Chart[:, cc])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wProp_Chart[:, cc] = wProp_Chart[:, cc] .* sf
        
        elseif propTot < propW_min
            sf = propW_min ./ propTot
            wProp_Chart[:, cc] = wProp_Chart[:, cc] .* sf
        end

        wInvest_Chart[:, cc] = wealth_Chart[cc, t-1] * wProp_Chart[:, cc]

    end

    totalDemand = sum((wInvest_Fund ./ assetPrice), dims = 2) + 
                  sum((wInvest_Chart ./ assetPrice), dims = 2)

    excessDemand = totalDemand .- assetSupply_max

    if any(x -> x != 0, excessDemand)
        return 1
    else
        return 0
    end 
end

