##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim
using PrettyTables

### Parameters

T = 1500             # Number of Timesteps
N = 3               # Number of Risky Assets
kChart = 15         # Number of Chartists
kFund = 15          # Number of Fundamentalists

phi = 0.002         # Dividend Growth Rate
phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
r = 0.0012          # Risk Free Rate
lambda = 3          # Relative Risk Aversion

wind_max = 96       # Max Exponential Moving Average Periods
wind_min = 24       # Min Exponential Moving Average Periods

meanR_max = 1.00     # Max Mean Reversion
meanR_min = 0.25     # Min Mean Reversion

corr_max = 0.60      # Max Expected Correlation Coefficient
corr_min = -0.60     # Min Expected Correlation Coefficient

propW_max = 0.95    # Max Wealth Investment Proportion
propW_min = -0.95   # Min Wealth Investment Proportion 

stock_max = 10      # Max Stock Position
stock_min = -5      # Min Stock Position

### Initialise Variables

div_0 = 0.002                                   # Initial Dividend
fund_0 = 2                                      # Initial Fundamental Value
wealth_0_Fund = ((N + 1) * fund_0) * 30          # Initial Fundamentalist Wealth
wealth_0_Chart = ((N + 1) * fund_0) * 10         # Initial Chartist Wealth

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Price Returns of Risky Assets
asset_Returns = zeros(N, T)     # Total Returns of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = ones(N, N, T, kFund)           # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = ones(N, N, T, kChart)         # Expected Return Covariance Array for Chartists

fill!(expRet_CovMat_Fund, 10)
fill!(expRet_CovMat_Chart, 0.5)

expPriceChange_Fund = zeros(N, T, kFund)            # Fundamentalists Expected Price Change of Risky Assets
expPriceReturn_Chart = zeros(N, T, kChart)          # Chartists Expected Price Return of Risky Assets

for i in 1:N
    dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
    fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
    price[i, 1] = fund_0 * 0.25     # Set Initial Asset Price
end

# Set Seed for Reproducibility
Random.seed!(1234)

# Fundamentalists Mean Reversion Parameter
meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

# Agent's Exponential Moving Average Period
ema_wind_Fund = rand(wind_min:4:wind_max, kFund)
ema_wind_Chart = rand(wind_min:4:wind_max, kChart)

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

    # Fundamentalists Exponential Moving Average Parameter
    ema_f = exp(-1/ema_wind_Fund[k])

    for ii in 1:N
        
        # Set Initial Portfolio Weights
        # wealthProp_Fund[ii, 1, k] = 1/(1 + N)   
        wealthProp_Fund[ii, 1, k] = 0.05    
        wealthInvest_Fund[ii, 1, k] = wealth_0_Fund * wealthProp_Fund[ii, 1, k] 

        # Set Initial Asset Demand 
        demand_Fund[ii, 1, k] = wealthInvest_Fund[ii, 1, k] / price[ii, 1]            

        expPriceChange_Fund[ii, 1, k] = (phi * fund_val[ii, 1]) + 
                                        (meanR[k] * (fund_val[ii, 1] - price[ii, 1]))
        expRet_Fund[ii, 1, k] = ((expPriceChange_Fund[ii, 1, k] + 
                                ((1 + phi) * dividends[ii, 1])) / price[ii, 1]) 
        expRet_CovMat_Fund[ii, ii, 1, k] = (ema_f * expRet_CovMat_Fund[ii, ii, 1, k])
    end

    expRet_CovMat_Fund[:, :, 1, k] = getCovMat(expRet_CovMat_Fund[:, :, 1, k], corr_coef_Fund[k, :])

    # Set Initial Risk-Free Prop weight 
    wealthProp_RF_Fund[k, 1] = (1 - sum(wealthProp_Fund[:, 1, k]))     
    wealthInvest_RF_Fund[k, 1] = wealth_0_Fund * wealthProp_RF_Fund[k, 1]

    # Set Initial Wealth of Fundamentalists

    wealth_Fund[k, 1] = wealthInvest_RF_Fund[k, 1] + 
                        sum(wealthInvest_Fund[:, 1, k])   
end

for k in 1:kChart

    # Chartists Exponential Moving Average Parameter
    ema_c = exp(-1/ema_wind_Chart[k])

    for ii in 1:N
        
        # Set Initial Portfolio Weights
        # wealthProp_Chart[ii, 1, k] = 1/(1 + N)
        wealthProp_Chart[ii, 1, k] = 0.05      
        wealthInvest_Chart[ii, 1, k] = wealth_0_Chart * wealthProp_Chart[ii, 1, k]
        
        # Set Initial Asset Demand
        demand_Chart[ii, 1, k] = wealthInvest_Chart[ii, 1, k] / price[ii, 1]            

        expPriceReturn_Chart[ii, 1, k] = (ema_c * 0.0015)
        expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] + 
                                (((1 + phi) * dividends[ii, 1]) / price[ii, 1])

        expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
    end

    expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])

    # Set Initial Risk-Free Prop weight
    wealthProp_RF_Chart[k, 1] = (1 - sum(wealthProp_Chart[:, 1, k]))       
    wealthInvest_RF_Chart[k, 1] = wealth_0_Chart * wealthProp_RF_Chart[k, 1]

    # Set Initial Wealth of Chartists

    wealth_Chart[k, 1] = wealthInvest_RF_Chart[k, 1] + 
                         sum(wealthInvest_Chart[:, 1, k])          
end

# Initialise Max Supply of each Risky Asset
assetSupply_max = sum(demand_Fund[1, 1, :]) + sum(demand_Chart[1, 1, :])

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
        elseif propTot < propW_min
            sf = propW_min ./ propTot
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
        elseif propTot < propW_min
            sf = propW_min ./ propTot
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
        elseif propTot < propW_min
            sf = propW_min ./ propTot
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
        elseif propTot < propW_min
            sf = propW_min ./ propTot
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

# Plot Check Asset Returns

function plotReturns(Returns, Time, kF, kC)

    t = 1:Time

    if N == 2

        p1 = plot(t, Returns[1, :], label = "Returns", title = "Asset 1, KF = $kF, KC = $kC", 
              xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p2 = plot(t, Returns[2, :], label = "Returns", title = "Asset 2, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)
                
        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        plot(p1, p2, layout = (N, 1), size = (800, 800))

    elseif N == 3

        p1 = plot(t, Returns[1, :], label = "Returns", title = "Asset 1, KF = $kF, KC = $kC", 
              xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p2 = plot(t, Returns[2, :], label = "Returns", title = "Asset 2, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)
                
        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p3 = plot(t, Returns[3, :], label = "Returns", title = "Asset 3, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        plot(p1, p2, p3, layout = (N, 1), size = (800, 800))

    elseif N == 4

        p1 = plot(t, Returns[1, :], label = "Returns", title = "Asset 1, KF = $kF, KC = $kC", 
              xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p2 = plot(t, Returns[2, :], label = "Returns", title = "Asset 2, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)
                
        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p3 = plot(t, Returns[3, :], label = "Returns", title = "Asset 3, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p4 = plot(t, Returns[4, :], label = "Returns", title = "Asset 4, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        plot(p1, p2, p3, p4, layout = (N, 1), size = (800, 800))

    elseif N == 5

        p1 = plot(t, Returns[1, :], label = "Returns", title = "Asset 1, KF = $kF, KC = $kC", 
              xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p2 = plot(t, Returns[2, :], label = "Returns", title = "Asset 2, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)
                
        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p3 = plot(t, Returns[3, :], label = "Returns", title = "Asset 3, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p4 = plot(t, Returns[4, :], label = "Returns", title = "Asset 4, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        p5 = plot(t, Returns[5, :], label = "Returns", title = "Asset 5, KF = $kF, KC = $kC", 
                xlabel = "T", ylabel = "Returns", legend = :topleft)

        hline!([0.00], label = false, color =:black, lw = 1, linestyle =:dash)

        plot(p1, p2, p3, p4, p5, layout = (N, 1), size = (800, 800))

    end

end

display(plotReturns(asset_Returns, T, kFund, kChart))

# Plot Check Price

function plotPrices(Prices, FValue, Time, kF, kC)

    t = 1:Time

    sz = 250 * N

    if N == 2

        p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
              Gamma = [$meanR_min, $meanR_max], Rho = [$corr_min, $corr_max], 
              Tau = [$propW_min, $propW_max], Stock = [$stock_min, $stock_max],
              EMA = [$wind_min, $wind_max]", 
              xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[1, t], 
            label = "Fundamental Value", linecolor=:red)

        p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[2, t], 
            label = "Fundamental Value", linecolor=:red)

        plot(p1, p2, layout = (N, 1), size = (800, sz))

    elseif N == 3

        p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
              Gamma = [$meanR_min, $meanR_max], Rho = [$corr_min, $corr_max], 
              Tau = [$propW_min, $propW_max], Stock = [$stock_min, $stock_max],
              EMA = [$wind_min, $wind_max]", 
              xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[1, t], 
            label = "Fundamental Value", linecolor=:red)

        p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[2, t], 
            label = "Fundamental Value", linecolor=:red)

        p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[3, t], 
            label = "Fundamental Value", linecolor=:red)

        plot(p1, p2, p3, layout = (3, 1), size = (800, sz))

    elseif N == 4

        p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
              Gamma = [$meanR_min, $meanR_max], Rho = [$corr_min, $corr_max], 
              Tau = [$propW_min, $propW_max], Stock = [$stock_min, $stock_max],
              EMA = [$wind_min, $wind_max]", 
              xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[1, t], 
            label = "Fundamental Value", linecolor=:red)

        p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[2, t], 
            label = "Fundamental Value", linecolor=:red)

        p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[3, t], 
            label = "Fundamental Value", linecolor=:red)

        p4 = plot(t, Prices[4, t], label = "Price", title = "Asset 4", 
            xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[4, t], 
              label = "Fundamental Value", linecolor=:red)

        plot(p1, p2, p3, p4, layout = (N, 1), size = (800, sz))

    elseif N == 5

        p1 = plot(t, Prices[1, :], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
              Gamma = [$meanR_min, $meanR_max], Rho = [$corr_min, $corr_max], 
              Tau = [$propW_min, $propW_max], Stock = [$stock_min, $stock_max],
              EMA = [$wind_min, $wind_max]", 
              xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[1, t], 
            label = "Fundamental Value", linecolor=:red)

        p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[2, t], 
            label = "Fundamental Value", linecolor=:red)

        p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3", 
                xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[3, t], 
            label = "Fundamental Value", linecolor=:red)

        p4 = plot(t, Prices[4, t], label = "Price", title = "Asset 4", 
            xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[4, t], 
              label = "Fundamental Value", linecolor=:red)

        p5 = plot(t, Prices[5, t], label = "Price", title = "Asset 5", 
            xlabel = "T", ylabel = "Price", legend = :topleft)

        plot!(t, FValue[5, t], 
              label = "Fundamental Value", linecolor=:red)
        plot(p1, p2, p3, p4, p5, layout = (N, 1), size = (800, sz))

    end

end

display(plotPrices(price, fund_val, T, kFund, kChart))

# Print Output from Model 

function printOutput(bt, et, agent, type)

    head = ["$bt", "$bt+1", "$bt+2", "$bt+3", "$bt+4", "$et-4", "$et-3", "$et-2", "$et-1", "$et"]

    if type == "ER"
        println("Fundamentalists Expected Return")
        pretty_table(expRet_Fund[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)
        println("Chartists Expected Return")
        pretty_table(expRet_Chart[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)

    elseif type == "Prop"
        println("Fundamentalists Proportion")
        pretty_table(wealthProp_Fund[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)
        println("Fundamentalists RF Proportion")
        pretty_table(transpose(wealthProp_RF_Fund[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Proportion")
        pretty_table(wealthProp_Chart[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)
        println("Chartists RF Proportion")
        pretty_table(transpose(wealthProp_RF_Chart[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
        header = head)

    elseif type == "Invest"
        println("Fundamentalists Wealth Invested")
        pretty_table(wealthInvest_Fund[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)
        println("Fundamentalists RF Wealth Invested")
        pretty_table(transpose(wealthInvest_RF_Fund[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Wealth Invested")
        pretty_table(wealthInvest_Chart[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et], agent],
                    header = head)
        println("Chartists RF Wealth Invested")
        pretty_table(transpose(wealthInvest_RF_Chart[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
    
    elseif type == "Wealth"
        println("Fundamentalists Wealth")
        pretty_table(transpose(wealth_Fund[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Wealth")
        pretty_table(transpose(wealth_Chart[agent, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)

    elseif type == "Price"
        println("Price")
        pretty_table(price[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]],
                    header = head)
        println("Price Return")
        pretty_table(price_returns[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]],
                    header = head)

    elseif type == "Demand"
        println("Fundamentalists Demand")
        df = reshape(sum(demand_Fund, dims = 3), 3, T)
        pretty_table(df[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]],
                    header = head)
        println("Chartists Demand")
        dc = reshape(sum(demand_Chart, dims = 3), 3, T)
        pretty_table(dc[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]],
                    header = head)
        println("Total Demand")
        dt = reshape(sum(demand_Fund, dims = 3) .+ sum(demand_Chart, dims = 3), 3, T)
        pretty_table(dt[:, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]],
                    header = head)
        println("Excess Demand")
        pretty_table(transpose(sqrt.(excessDemand_Optim[1, [bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]])),
                    header = head)
    end
end

printOutput(1, T, 5, "ER")
printOutput(1, T, 5, "Prop")
printOutput(1, T, 5, "Invest")
printOutput(1, T, 5, "Wealth")
printOutput(1, T, 5, "Price")
printOutput(1, T, 5, "Demand")