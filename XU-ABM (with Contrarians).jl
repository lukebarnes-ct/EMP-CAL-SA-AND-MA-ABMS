##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions
using Optim

### Parameters

T = 100             # Number of Timesteps
N = 3               # Number of Risky Assets
kChart = 10         # Number of Chartists
kCont = 10          # Number of Contrarians

phi = 0.001         # Dividend Growth Rate
phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
r = 0.0012          # Risk Free Rate
lambda = 3          # Relative Risk Aversion

wind_max = 80       # Max Exponential Moving Average Periods
wind_min = 20       # Min Exponential Moving Average Periods

corr_max = 0.8      # Max Expected Correlation Coefficient
corr_min = -0.2     # Min Expected Correlation Coefficient

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

assetSupply_max = (kCont * asset_0 * supplyMult) + (kChart * asset_0 * supplyMult)       # Initialise Max Supply of each Risky Asset

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets
price_returns = zeros(N, T)     # Price Returns of Risky Assets
asset_Returns = zeros(N, T)     # Total Returns of Risky Assets

expRet_Cont= zeros(N, T, kCont)                    # Contrarians Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                 # Chartists Expected Return of Risky Assets
expRet_CovMat_Cont= ones(N, N, T, kCont)           # Expected Return Covariance Array for Contrarians
expRet_CovMat_Chart = ones(N, N, T, kChart)        # Expected Return Covariance Array for Chartists

fill!(expRet_CovMat_Cont, 1)
fill!(expRet_CovMat_Chart, 1)

expPriceReturn_Cont= zeros(N, T, kCont)             # Contrarians Expected Price Return of Risky Assets
expPriceReturn_Chart = zeros(N, T, kChart)          # Chartists Expected Price Return of Risky Assets

for i in 1:N
    dividends[i, 1] = div_0         # Set Initial Dividend in Matrix
    fund_val[i, 1] = fund_0         # Set Initial Fundamental Value
    price[i, 1] = fund_0 * 1.00     # Set Initial Asset Price
end

Random.seed!(1234)                  # Set Seed for Reproducibility

# Agent's Exponential Moving Average Period
ema_wind_Cont= rand(wind_min:wind_max, kCont)
ema_wind_Chart = rand(wind_min:wind_max, kChart)

# Agent's Expected Correlation Coefficients for the Risky Assets
triAg = floor(Int, (N * (N - 1)) / (2))
corr_coef_Cont= round.(rand(Uniform(corr_min, corr_max), 
                        kCont, triAg), digits = 2)

corr_coef_Chart = round.(rand(Uniform(corr_min, corr_max), 
                         kChart, triAg), digits = 2)

wealth_Cont = zeros(kCont, T)              # Contrarians Wealth
wealth_Chart  = zeros(kChart, T)            # Chartists Wealth

wealthProp_Cont= zeros(N, T, kCont)        # Contrarians Proportion of Wealth Invested in Risky Assets
wealthProp_Chart = zeros(N, T, kChart)      # Chartists Proportion of Wealth Invested in Risky Assets

wealthProp_RF_Cont= zeros(kCont, T)        # Contrarians Proportion of Wealth Invested in Risk-Free Asset
wealthProp_RF_Chart = zeros(kChart, T)      # Chartists Proportion of Wealth Invested in Risk-Free Asset

wealthInvest_Cont= zeros(N, T, kCont)      # Contrarians Wealth Invested in Risky Assets
wealthInvest_Chart = zeros(N, T, kChart)    # Chartists Wealth Invested in Risky Assets

wealthInvest_RF_Cont= zeros(kCont, T)      # Contrarians Wealth Invested in Risk-Free Asset
wealthInvest_RF_Chart = zeros(kChart, T)    # Chartists Wealth Invested in Risk-Free Asset

demand_Cont= zeros(N, T, kCont)            # Contrarians Demand of Risky Assets
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

for k in 1:kCont
    wealthProp_RF_Cont[k, 1] = 1/(1 + N)           # Set Initial Risk-Free Prop weight
    wealthInvest_RF_Cont[k, 1] = cash_0

    # Contrarians Exponential Moving Average Parameter
    ema_f = exp(-1/ema_wind_Cont[k])

    for ii in 1:N
        wealthProp_Cont[ii, 1, k] = 1/(1 + N)       # Set Initial Portfolio Weights
        wealthInvest_Cont[ii, 1, k] = cash_0

        demand_Cont[ii, 1, k] = asset_0             # Set Initial Asset Demand 

        expPriceReturn_Cont[ii, 1, k] = (ema_f * 0.25) 
        expRet_Cont[ii, 1, k] = -expPriceReturn_Cont[ii, 1, k] 
        expRet_CovMat_Cont[ii, ii, 1, k] = (ema_f * expRet_CovMat_Cont[ii, ii, 1, k])
    end

    expRet_CovMat_Cont[:, :, 1, k] = getCovMat(expRet_CovMat_Cont[:, :, 1, k], corr_coef_Cont[k, :])

    # Set Initial Wealth of Contrarians

    wealth_Cont[k, 1] = wealthInvest_RF_Cont[k, 1] + 
                        sum(wealthInvest_Cont[:, 1, k])   
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

        expPriceReturn_Chart[ii, 1, k] = (ema_c * 0.25) 
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
    
    eP_Return_Cont = zeros(N, 2, kCont)
    eP_Return_Cont[:, 1, :] = expPriceReturn_Cont[:, TT-1, :]
    eR_Cont= zeros(N, 2, kCont)
    eR_Cont[:, 1, :] = expRet_Cont[:, TT-1, :]
    eR_Cov_Cont= ones(N, N, 2, kCont)
    eR_Cov_Cont[:, :, 1, :] = expRet_CovMat_Cont[:, :, TT-1, :]
    pReturns = (assetPrice .- price[:, TT-1]) ./ price[:, TT-1]
    returns = pReturns .+ (dividends[:, TT] ./ price[:, TT-1])

    eP_Return = zeros(N, 2, kChart)
    eP_Return[:, 1, :] = expPriceReturn_Chart[:, TT-1, :]
    eR_Chart = zeros(N, 2, kChart)
    eR_Chart[:, 1, :] = expRet_Chart[:, TT-1, :]
    eR_Cov_Chart = ones(N, N, 2, kChart)
    eR_Cov_Chart[:, :, 1, :] = expRet_CovMat_Chart[:, :, TT-1, :]

    wProp_Cont= zeros(N, kCont)
    wProp_Chart = zeros(N, kChart)
    wInvest_Cont= zeros(N, kCont)
    wInvest_Chart = zeros(N, kChart)

    d_Cont= zeros(N, kCont, 2)
    d_Cont[:, :, 1] = demand_Cont[:, TT-1, :]
    d_Chart = zeros(N, kChart, 2)
    d_Chart[:, :, 1] = demand_Chart[:, TT-1, :]

    for i in 1:N

        for f in 1:kCont

            # Contrarians Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Cont[f])

            eP_Return_Cont[i, 2, f] = (ema_f * eP_Return_Cont[i, 1, f]) + 
                                      ((1 - ema_f) * -pReturns[i])

            # Contrarians Expected Return for the i-th Asset at time t
            eR_Cont[i, 2, f] =  (eP_Return_Cont[i, 2, f] + 
                                (((1 + phi) * dividends[i, TT])/assetPrice[i]))
        
            # Diagonal of Contrarians Covariance Matrix of Expected Returns at time t
            eR_Cov_Cont[i, i, 2, f] = (ema_f * eR_Cov_Cont[i, i, 1, f]) + 
                                      ((1 - ema_f) * (eR_Cont[i, 1, f] - returns[i])^2)

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

    for ff in 1:kCont

        # Contrarians Covariance Matrix of Expected Returns at time t
        eR_Cov_Cont[:, :, 2, ff] = getCovMat(eR_Cov_Cont[:, :, 2, ff], corr_coef_Cont[ff, :])

        # Contrarians Portfolio of Risky Assets
        wProp_Cont[:, ff] = (1/lambda) * inv(eR_Cov_Cont[:, :, 2, ff]) * 
                            (eR_Cont[:, 2, ff] .- r)

        wProp_Cont[:, ff] = min.(max.(wProp_Cont[:, ff], propW_min), propW_max)

        # Use Proportional Scaling if conditions violated

        propTot = sum(wProp_Cont[:, ff])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wProp_Cont[:, ff] = wProp_Cont[:, ff] .* sf
        end

        wInvest_Cont[:, ff] = wealth_Cont[ff, TT-1] * wProp_Cont[:, ff]

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

    d_Cont[:, :, 2] = wInvest_Cont./ assetPrice
    d_Chart[:, :, 2] = wInvest_Chart ./ assetPrice

    for i in 1:N

        for f in 1:kCont

            dem = d_Cont[i, f, 2]

            if dem > stock_max

                d_Cont[i, f, 2] = stock_max

            elseif dem < stock_min

                d_Cont[i, f, 2] = stock_min

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

    totalDemand = sum((d_Cont[:, :, 2]), dims = 2) + 
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

    excessDemand_Optim[1, t] = Optim.minimum(resOpt)

    # Calculate Price Returns
    price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

    # Calculate Asset Returns
    asset_Returns[:, t] = price_returns[:, t] .+ (dividends[:, t] ./ price[:, t-1])

    for i in 1:N

        for f in 1:kCont

            # Contrarians Exponential Moving Average Parameter
            ema_f = exp(-1/ema_wind_Cont[f])

            expPriceReturn_Cont[i, t, f] = (ema_f * expPriceReturn_Cont[i, t-1, f]) + 
                                           ((1 - ema_f) * -(price_returns[i, t]))

            # Contrarians Expected Return for the i-th Asset at time t
            expRet_Cont[i, t, f] = (expRet_Cont[i, t, f] = expPriceReturn_Cont[i, t, f] + 
                                   (((1 + phi) * dividends[i, t])/price[i, t]))
        
            # Diagonal of Contrarians Covariance Matrix of Expected Returns at time t
            expRet_CovMat_Cont[i, i, t, f] = (ema_f * expRet_CovMat_Cont[i, i, t-1, f]) + 
                                             ((1 - ema_f) * (expRet_Cont[i, t-1, f] - price_returns[i, t])^2)

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

    for ff in 1:kCont

        # Contrarians Covariance Matrix of Expected Returns at time t
        expRet_CovMat_Cont[:, :, t, ff] = getCovMat(expRet_CovMat_Cont[:, :, t, ff], corr_coef_Cont[ff, :])

        # Contrarians Portfolio of Risky Assets
        wealthProp_Cont[:, t, ff] = (1/lambda) * inv(expRet_CovMat_Cont[:, :, t, ff]) * (expRet_Cont[:, t, ff] .- r)

        # Ensure Contrarians Portfolio does not violate max/min Conditions

        wealthProp_Cont[:, t, ff] = min.(max.(wealthProp_Cont[:, t, ff], propW_min), propW_max)

        # Use Proportional Scaling if conditions violated

        propTot = sum(wealthProp_Cont[:, t, ff])

        if propTot > propW_max
            sf = propW_max ./ propTot
            wealthProp_Cont[:, t, ff] = wealthProp_Cont[:, t, ff] .* sf
        end

        wealthInvest_Cont[:, t, ff] = wealth_Cont[ff, t-1] * wealthProp_Cont[:, t, ff]

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
    demand_Cont[:, t, :] = (wealthInvest_Cont[:, t, :]) ./ price[:, t]
    demand_Chart[:, t, :] = (wealthInvest_Chart[:, t, :]) ./ price[:, t]

    for i in 1:N

        for f in 1:kCont

            dem = demand_Cont[i, t, f]

            if dem > stock_max

                demand_Cont[i, t, f] = stock_max

            elseif dem < stock_min

                demand_Cont[i, t, f] = stock_min

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

        for f in 1:kCont

            wealthInvest_Cont[i, t, f] = demand_Cont[i, t, f] * price[i, t]
            wealthProp_Cont[i, t, f] = wealthInvest_Cont[i, t, f] / wealth_Cont[f, t-1]
        end

        for c in 1:kChart

            wealthInvest_Chart[i, t, c] = demand_Chart[i, t, c] * price[i, t]
            wealthProp_Chart[i, t, c] = wealthInvest_Chart[i, t, c] / wealth_Chart[c, t-1]
        end

    end

    # Update Contrarians Investment in the Risk-Free Asset
    wealthProp_RF_Cont[:, t] = (1 .- sum(wealthProp_Cont[:, t, :], dims = 1))
    wealthInvest_RF_Cont[:, t] = wealth_Cont[:, t-1] .* wealthProp_RF_Cont[:, t]

    # Update Chartists Investment in the Risk-Free Asset
    wealthProp_RF_Chart[:, t] = (1 .- sum(wealthProp_Chart[:, t, :], dims = 1))
    wealthInvest_RF_Chart[:, t] = wealth_Chart[:, t-1] .* wealthProp_RF_Chart[:, t]

    # Update Contrarians Wealth at Market Clearing Prices
    wealth_Cont[:, t] = transpose(wealthInvest_RF_Cont[:, t] .* (1 + r)) + 
                        (sum(wealthInvest_Cont[:, t, :] .* 
                        ((price[:, t] + dividends[:, t]) ./ (price[:, t-1])), dims = 1))

    wealth_Cont[:, t] = round.(wealth_Cont[:, t], digits = 2)

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
expRet_Cont[:, b_ttt:e_ttt, fff]
expRet_CovMat_Cont[:, :, b_ttt:e_ttt, fff]
expRet_Chart[:, b_ttt:e_ttt, ccc]
expRet_CovMat_Chart[:, :, b_ttt:e_ttt, ccc]

# Checks (3)

wealthProp_Cont[:, b_ttt:e_ttt, fff]
wealthProp_RF_Cont[fff, b_ttt:e_ttt]
wealthInvest_Cont[:, b_ttt:e_ttt, fff]
wealthInvest_RF_Cont[fff, b_ttt:e_ttt]
wealth_Cont[fff, b_ttt:e_ttt]

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

demand_Cont[:, b_ttt:e_ttt, fff]
demand_Chart[:, b_ttt:e_ttt, ccc]

sum(demand_Cont, dims = 3)
sum(demand_Chart, dims = 3)

sum(demand_Cont, dims = 3) .+ sum(demand_Chart, dims = 3)

excessDemand_Optim

# Plot Check Price

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

# Plot Check Asset Returns

p4 = plot(b_ttt:e_ttt, asset_Returns[1, b_ttt:e_ttt], label = "Returns", title = "Asset 1", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

p5 = plot(b_ttt:e_ttt, asset_Returns[2, b_ttt:e_ttt], label = "Returns", title = "Asset 2", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

p6 = plot(b_ttt:e_ttt, asset_Returns[3, b_ttt:e_ttt], label = "Returns", title = "Asset 3", 
          xlabel = "T", ylabel = "Returns", legend = :topright)

plot(p4, p5, p6, layout = (3, 1), size = (800, 800))

# Checks (7)

all(wealth_Cont.>= 0)
all(wealth_Chart .>= 0)

# Checks (8)

sum((price .- fund_val).^2)