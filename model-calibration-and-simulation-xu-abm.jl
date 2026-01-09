##### Manual Model Calibration for XU ABM

using Random
using Plots
using Distributions
using Optim
using NLsolve
using NLopt
using JLD2
using Base.Threads
using Printf
using Plots.PlotMeasures
using StatsBase
using TypedTables
using StatsPlots
using Subscripts
using HypothesisTests
using Hurst
using DataFrames

# Load JSE Top 40 Index and Log Returns from .jld2 file
@load "Data/jsetop40_weekly.jld2" weekly_JSETOP40_Data
@load "Data/jsetop40_daily.jld2" daily_JSETOP40_Data
@load "Data/jsetop40_weekly_LogReturns.jld2" returnsJSE_Weekly
@load "Data/jsetop40_daily_LogReturns.jld2" returnsJSE_Daily

# Load SSE 50 Index and Log Returns from .jld2 file
@load "Data/sse50_weekly.jld2" weekly_SSE50_Data
@load "Data/sse50_daily.jld2" daily_SSE50_Data
@load "Data/sse50_weekly_LogReturns.jld2" returnsSSE50_Weekly
@load "Data/sse50_daily_LogReturns.jld2" returnsSSE50_Daily

# Load BSE Sensex Index and Log Returns from .jld2 file
@load "Data/bsesn_weekly.jld2" weekly_BSESN_Data
@load "Data/bsesn_daily.jld2" daily_BSESN_Data
@load "Data/bsesn_weekly_LogReturns.jld2" returnsBSESN_Weekly
@load "Data/bsesn_daily_LogReturns.jld2" returnsBSESN_Daily

lengthJSE_Daily = length(daily_JSETOP40_Data)
lengthJSE_Weekly = length(weekly_JSETOP40_Data)

lengthSSE50_Daily = length(daily_SSE50_Data)
lengthSSE50_Weekly = length(weekly_SSE50_Data)

lengthBSESN_Daily = length(daily_BSESN_Data)
lengthBSESN_Weekly = length(weekly_BSESN_Data)

#####################################################################
# Xu et al. ABM

function xuABM(Time, n, N, 
               kC, kF,
               w_max_Fund, w_min_Fund, 
               w_max_Chart, w_min_Chart)

    ### Parameters

    T = Time            # Number of Timesteps
    N = N               # Number of Risky Assets

    ### Hyperparameters

    kChart = kC         # Number of Chartists
    kFund = kF          # Number of Fundamentalists

    wind_max_Fund = w_max_Fund       # Fundamentalists Max Exponential Moving Average Periods
    wind_min_Fund = w_min_Fund       # Fundamentalists Min Exponential Moving Average Periods

    wind_max_Chart = w_max_Chart       # Chartists Max Exponential Moving Average Periods
    wind_min_Chart = w_min_Chart       # Chartists Min Exponential Moving Average Periods

    ### Parameters

    phi = 0.0025        # Dividend Growth Rate
    phi_sd = 0.0125     # Dividend Growth Rate Standard Deviation
    r = 0.0012          # Risk Free Rate
    lambda = 3          # Relative Risk Aversion

    meanR_max = 1.00     # Max Mean Reversion
    meanR_min = 0.00     # Min Mean Reversion

    corr_max = 0.80       # Max Expected Correlation Coefficient
    corr_min = -0.60      # Min Expected Correlation Coefficient

    propW_max = 0.95      # Max Wealth Investment Proportion
    propW_min = -0.95     # Min Wealth Investment Proportion 

    stock_max = 2        # Max Stock Position
    stock_min = -4       # Min Stock Position

    ### Initialise Variables

    div_0 = 0.002                                   # Initial Dividend
    fund_0 = 10                                     # Initial Fundamental Value
    wealth_0_Fund = (N + 1) * 48          # Initial Fundamentalist Wealth
    wealth_0_Chart = (N + 1) * 10         # Initial Chartist Wealth

    dividends = zeros(N, T)         # Dividends of Risky Assets
    fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
    price = zeros(N, T)             # Prices of Risky Assets
    price_returns = zeros(N, T)     # Price Returns of Risky Assets
    asset_Returns = zeros(N, T)     # Total Returns of Risky Assets
    log_returns = zeros(N, T)       # Log Returns of Risky Assets

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
    end

    if N == 2

        price[:, 1] = [fund_0 * 0.55, fund_0 * 0.65]

    elseif N == 3

        price[:, 1] = [fund_0 * 0.55, fund_0 * 0.65, fund_0 * 0.51]

    elseif N == 4

        price[:, 1] = [fund_0 * 0.55, fund_0 * 0.65, fund_0 * 0.51, fund_0 * 0.51]

    elseif N == 5

        price[:, 1] = [fund_0 * 0.55, fund_0 * 0.65, fund_0 * 0.51, fund_0 * 0.51, fund_0 * 0.51]

    else

        price[:, 1] = fund_val[:, 1] .* 0.5
    end

    # Set Seed for Reproducibility
    Random.seed!(n)

    # Fundamentalists Mean Reversion Parameter
    meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)

    # Agent's Exponential Moving Average Period
    ema_wind_Fund = rand(wind_min_Fund:25:wind_max_Fund, kFund)
    ema_wind_Chart = rand(wind_min_Chart:2:wind_max_Chart, kChart)

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
            wealthProp_Fund[ii, 1, k] = 1/(1 + N)
            wealthInvest_Fund[ii, 1, k] = (wealth_0_Fund/3.2) * wealthProp_Fund[ii, 1, k] 
    
            # Set Initial Asset Demand 
            demand_Fund[ii, 1, k] = wealthInvest_Fund[ii, 1, k] / fund_val[ii, 1]            
    
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
    
        wealth_Fund[k, 1] = wealth_0_Fund   
    end
    
    for k in 1:kChart
    
        # Chartists Exponential Moving Average Parameter
        ema_c = exp(-1/ema_wind_Chart[k])
    
        for ii in 1:N
            
            # Set Initial Portfolio Weights
            wealthProp_Chart[ii, 1, k] = 1/(1 + N)
            wealthInvest_Chart[ii, 1, k] = wealth_0_Chart * wealthProp_Chart[ii, 1, k]
            
            # Set Initial Asset Demand
            demand_Chart[ii, 1, k] = wealthInvest_Chart[ii, 1, k] / fund_val[ii, 1]            
    
            expPriceReturn_Chart[ii, 1, k] = (ema_c * 0.01)
            expRet_Chart[ii, 1, k] = expPriceReturn_Chart[ii, 1, k] + 
                                    (((1 + phi) * dividends[ii, 1]) / price[ii, 1])
    
            expRet_CovMat_Chart[ii, ii, 1, k] = (ema_c * expRet_CovMat_Chart[ii, ii, 1, k])
        end
    
        expRet_CovMat_Chart[:, :, 1, k] = getCovMat(expRet_CovMat_Chart[:, :, 1, k], corr_coef_Chart[k, :])
    
        # Set Initial Risk-Free Prop weight
        wealthProp_RF_Chart[k, 1] = (1 - sum(wealthProp_Chart[:, 1, k]))       
        wealthInvest_RF_Chart[k, 1] = wealth_0_Chart * wealthProp_RF_Chart[k, 1]
    
        # Set Initial Wealth of Chartists
    
        wealth_Chart[k, 1] = wealth_0_Chart        
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
        
        if TT % 50 == 0
            println("Time is: ", TT)
        end

        err = rand(Normal(0, 1), N)                                             # Standard Normal Error Term
        dividends[:, t] = (1 + phi .+ phi_sd * err) .* dividends[:, t-1]        # Expected Dividends for Next Time Period
        fund_val[:, t] = (1 + phi .+ phi_sd * err) .* fund_val[:, t-1]          # Expected Fundamental Value for Next Time Period
        
        resPrice = price[:, t-1]

        resOpt = Optim.optimize(optDemand, resPrice, NelderMead())

        # Determine the price that will Clear each market of Risky Assets
        price[:, t] = Optim.minimizer(resOpt)

        # Calculate Price Returns
        price_returns[:, t] = ((price[:, t] - price[:, t-1]) ./ price[:, t-1])

        # Calculate Log Returns
        log_returns[:, t] = log.(price[:, t] ./ price[:, t-1])

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

    return price, log_returns

end

function xuIndex(N, prices)
    
    indexLength = size(prices, 2)
    indexPrice = sum((1/N) * prices, dims = 1)

    indexReturns = zeros(indexLength)

    for i in 2:indexLength

        indexReturns[i] = log(indexPrice[i]/indexPrice[i-1])

    end

    return indexPrice, indexReturns
end

#####################################################################

function teststatistic(x)
    n = x.n_x*x.n_y/(x.n_x+x.n_y)
    sqrt(n)*x.δ
end

function getMoments(Returns, bt, et, type, index, timescale)

    t = bt:et

    U = zeros(6)

    U[1] = round(mean(Returns[t]), digits = 4)

    U[2] = round(std(Returns[t]), digits = 4)

    U[3] = round(skewness(Returns[t]), digits = 4)

    U[4] = round(kurtosis(Returns[t]), digits = 4)

    if type == "Simulated"

        if index == "JSE"

            if timescale == "Daily"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsJSE_Daily)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            elseif timescale == "Weekly"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsJSE_Weekly)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            end
    
        elseif index == "SSE"
    
            if timescale == "Daily"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsSSE50_Daily)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            elseif timescale == "Weekly"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsSSE50_Weekly)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            end
    
        elseif index == "BSE"
    
            if timescale == "Daily"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsBSESN_Daily)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            elseif timescale == "Weekly"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsBSESN_Weekly)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            end
    
        end

    else

        U[5] = 0
    end

    U[6] = round(hurst_exponent(Returns[t], 1:19)[1], digits = 4)

    return U

end

function getSimulatedMoments(par, N, index, timescale)

    timeBegin = 101
    timeEnd = 600

    sMoments = zeros(6, N)

    KC = par[1]
    KF = par[2]
    WMAX_F = par[3]
    WMIN_F = par[4]
    WMAX_C = par[5]
    WMIN_C = par[6]

    NumAssets = 5

    for n in 1:N

        priceMatrix, returnMatrix = xuABM(timeEnd, n, NumAssets, KC, KF, WMAX_F, WMIN_F, WMAX_C, WMIN_C)
        xuIndexPrice, xuIndexReturn = xuIndex(NumAssets, priceMatrix)

        moments = getMoments(xuIndexReturn, timeBegin, timeEnd, "Simulated", index, timescale)

        sMoments[:, n] = moments

    end

    simMoments = mean(sMoments, dims = 2)

    return simMoments

end

function getBlockMoments(Returns, blockReturns)

    U = zeros(6)

    U[1] = round(mean(blockReturns), digits = 4)

    U[2] = round(std(blockReturns), digits = 4)

    U[3] = round(skewness(blockReturns), digits = 4)

    U[4] = round(kurtosis(blockReturns), digits = 4)

    ks_test = ApproximateTwoSampleKSTest(blockReturns, Returns)
    U[5] = round(teststatistic(ks_test), digits = 4)

    U[6] = round(hurst_exponent(blockReturns, 1:19)[1], digits = 4)

    return U

end

function getMovingBlockBootstrapMatrix(seed, Returns, b, s)

    # Set Seed for Reproducibility
    Random.seed!(seed)

    lenRet = length(Returns)
    numBlocks = ceil(Int, lenRet / b)
    blockStart = 1:(lenRet - b + 1)

    movingBlocks = [Returns[i:i + b - 1] for i in blockStart]

    bootSamples = zeros(lenRet, s)
    bootMoments = zeros(6, s)

    for boot in 1:s

        samInd = sample(blockStart, numBlocks, replace = true)
        samBlocks = [movingBlocks[i] for i in samInd]
        bootVect = vcat(map(collect, samBlocks)...)
        bootSamples[:, boot] = bootVect[1:lenRet]
        bootMoments[:, boot] = getBlockMoments(Returns, bootSamples[:, boot])

    end

    invSigmaU = inv(cov(bootMoments, dims = 2))

    return invSigmaU
end

function getObjectiveFunction(moments, simMoments, InvBootMatrix)

    momDiffVect = simMoments .- moments 

    objFunc = momDiffVect' * InvBootMatrix * momDiffVect

    return objFunc
end

function f_XU(par, repetitions, index, timescale)

    simMom = getSimulatedMoments(par, repetitions, index, timescale)

    if index == "JSE"

        if timescale == "Daily"
    
            obj = getObjectiveFunction(momentsJSE_Daily, simMom, bootstrapMatrixJSE_Daily)
    
        elseif timescale == "Weekly"
    
            obj = getObjectiveFunction(momentsJSE_Weekly, simMom, bootstrapMatrixJSE_Weekly)
    
        end
    
    elseif index == "SSE"
    
        if timescale == "Daily"
    
            obj = getObjectiveFunction(momentsSSE50_Daily, simMom, bootstrapMatrixSSE50_Daily)
    
        elseif timescale == "Weekly"
    
            obj = getObjectiveFunction(momentsSSE50_Weekly, simMom, bootstrapMatrixSSE50_Weekly)
    
        end
    
    elseif index == "BSE"
    
        if timescale == "Daily"
    
            obj = getObjectiveFunction(momentsBSESN_Daily, simMom, bootstrapMatrixBSESN_Daily)
    
        elseif timescale == "Weekly"
    
            obj = getObjectiveFunction(momentsBSESN_Weekly, simMom, bootstrapMatrixBSESN_Weekly)
    
        end
    
    end

    return obj[1]
end

#####################################################################

# Find the Required Moments for each of the Empirical Time Series 

momentsJSE_Daily = getMoments(returnsJSE_Daily, 1, lengthJSE_Daily, "Real", "JSE", "Daily")
momentsJSE_Weekly = getMoments(returnsJSE_Weekly, 1, lengthJSE_Weekly, "Real", "JSE", "Weekly")

momentsSSE50_Daily = getMoments(returnsSSE50_Daily, 1, lengthSSE50_Daily, "Real", "SSE50", "Daily")
momentsSSE50_Weekly = getMoments(returnsSSE50_Weekly, 1, lengthSSE50_Weekly, "Real", "SSE50", "Weekly")

momentsBSESN_Daily = getMoments(returnsBSESN_Daily, 1, lengthBSESN_Daily, "Real", "BSESN", "Daily")
momentsBSESN_Weekly = getMoments(returnsBSESN_Weekly, 1, lengthBSESN_Weekly, "Real", "BSESN", "Weekly")

# Find the Moving Block Bootstrap Matrix for each of the Empirical Log Return Time Series

blockWindow = 100
blockSamples = 1000

bootstrapMatrixJSE_Daily = getMovingBlockBootstrapMatrix(2001, returnsJSE_Daily, blockWindow, blockSamples)
bootstrapMatrixJSE_Weekly = getMovingBlockBootstrapMatrix(2002, returnsJSE_Weekly, blockWindow, blockSamples)

bootstrapMatrixSSE50_Daily = getMovingBlockBootstrapMatrix(2003, returnsSSE50_Daily, blockWindow, blockSamples)
bootstrapMatrixSSE50_Weekly = getMovingBlockBootstrapMatrix(2004, returnsSSE50_Weekly, blockWindow, blockSamples)

bootstrapMatrixBSESN_Daily = getMovingBlockBootstrapMatrix(2005, returnsBSESN_Daily, blockWindow, blockSamples)
bootstrapMatrixBSESN_Weekly = getMovingBlockBootstrapMatrix(2006, returnsBSESN_Weekly, blockWindow, blockSamples)

#####################################################################

function calibrateABM(index, timescale)

    maxWindFund = [100, 125, 150]
    minWindFund = [25, 50, 75]
    maxWindChart = [75, 100, 125]
    minWindChart = [15, 25, 35]

    param = [20, 20, 100, 25, 75, 15]

    bestOBJ = 1e8
    bestParam = param 

    output = DataFrame(MaxWindFund = Int[], MinWindFund = Int[], MaxWindChart = Int[], MinWindChart = Int[], Objective = Float64[])

    @time for a in maxWindFund

        param[3] = a

        @time for b in minWindFund

            param[4] = b

            @time for c in maxWindChart

                param[5] = c

                @time for d in minWindChart

                    param[6] = d

                    println("Maximum Window Fundamentalists: ", a, " Minimum Window Fundamentalists: ", b)
                    println("Maximum Window Chartists: ", c, " Minimum Window Chartists: ", d)

                    objFuncVal = f_XU(param, 3, index, timescale)

                    push!(output, (a, b, c, d, objFuncVal))
                    println(output)

                    if objFuncVal < bestOBJ

                        println("NEW BEST PARAMETERS: $param")
                        println("NEW BEST VALUE: $objFuncVal")

                        bestOBJ = objFuncVal
                        bestParam = param

                    end

                end
            end

        end

    end

    return bestOBJ, bestParam, output
end

#####################################################################

# Run the MSM Nelder Mead Optimisation for each Empirical Log Return Time Series

NumAssets = 5 
jseTime = 100 + lengthJSE_Weekly
sseTime = 100 + lengthSSE50_Weekly
bseTime = 100 + lengthBSESN_Weekly
id = 1

# JSE Weekly Log Returns 

bestObjective_JSE_Weekly, bestParameters_JSE_Weekly, output__JSE_Weekly = calibrateABM("JSE", "Weekly")
priceMatrix_XU_JSE_Weekly, logReturnsMatrix_XU_JSE_Weekly = xuABM(jseTime, id, NumAssets, 
                                                                  bestParameters_JSE_Weekly[1], bestParameters_JSE_Weekly[2], 
                                                                  bestParameters_JSE_Weekly[3], bestParameters_JSE_Weekly[4], 
                                                                  bestParameters_JSE_Weekly[5], bestParameters_JSE_Weekly[6])
xuIndexPrice_JSE_Weekly, xuIndexReturn_JSE_Weekly = xuIndex(NumAssets, priceMatrix_XU_JSE_Weekly[:, 101:jseTime])

bestParameters_JSE_Weekly = collect(output__JSE_Weekly[argmin(output__JSE_Weekly.Objective), 1:4])

@save "Data/xu-calibration/prices-jse-weekly.jld2" priceMatrix_XU_JSE_Weekly
@save "Data/xu-calibration/log-returns-jse-weekly.jld2" logReturnsMatrix_XU_JSE_Weekly
@save "Data/xu-calibration/parameters-jse-weekly.jld2" bestParameters_JSE_Weekly
@save "Data/xu-calibration/objective-results-jse-weekly.jld2" output__JSE_Weekly
@save "Data/xu-calibration/xu-index-price-jse-weekly.jld2" xuIndexPrice_JSE_Weekly
@save "Data/xu-calibration/xu-index-log-returns-jse-weekly.jld2" xuIndexReturn_JSE_Weekly

# SSE50 Weekly Log Returns 

bestObjective_SSE50_Weekly, bestParameters_SSE50_Weekly, output__SSE50_Weekly = calibrateABM("SSE", "Weekly")
priceMatrix_XU_SSE50_Weekly, logReturnsMatrix_XU_SSE50_Weekly = xuABM(sseTime, id, NumAssets, 
                                                                  bestParameters_SSE50_Weekly[1], bestParameters_SSE50_Weekly[2], 
                                                                  bestParameters_SSE50_Weekly[3], bestParameters_SSE50_Weekly[4], 
                                                                  bestParameters_SSE50_Weekly[5], bestParameters_SSE50_Weekly[6])
xuIndexPrice_SSE50_Weekly, xuIndexReturn_SSE50_Weekly = xuIndex(NumAssets, priceMatrix_XU_SSE50_Weekly[:, 101:sseTime])

bestParameters_SSE50_Weekly = collect(output__SSE50_Weekly[argmin(output__SSE50_Weekly.Objective), 1:4])

@save "Data/xu-calibration/prices-sse50-weekly.jld2" priceMatrix_XU_SSE50_Weekly
@save "Data/xu-calibration/log-returns-sse50-weekly.jld2" logReturnsMatrix_XU_SSE50_Weekly
@save "Data/xu-calibration/parameters-sse50-weekly.jld2" bestParameters_SSE50_Weekly
@save "Data/xu-calibration/objective-results-sse50-weekly.jld2" output__SSE50_Weekly
@save "Data/xu-calibration/xu-index-price-sse50-weekly.jld2" xuIndexPrice_SSE50_Weekly
@save "Data/xu-calibration/xu-index-log-returns-sse50-weekly.jld2" xuIndexReturn_SSE50_Weekly

# BSESN Weekly Log Returns 

bestObjective_BSESN_Weekly, bestParameters_BSESN_Weekly, output__BSESN_Weekly = calibrateABM("BSE", "Weekly")
priceMatrix_XU_BSESN_Weekly, logReturnsMatrix_XU_BSESN_Weekly = xuABM(bseTime, id, NumAssets, 
                                                                  bestParameters_BSESN_Weekly[1], bestParameters_BSESN_Weekly[2], 
                                                                  bestParameters_BSESN_Weekly[3], bestParameters_BSESN_Weekly[4], 
                                                                  bestParameters_BSESN_Weekly[5], bestParameters_BSESN_Weekly[6])
xuIndexPrice_BSESN_Weekly, xuIndexReturn_BSESN_Weekly = xuIndex(NumAssets, priceMatrix_XU_BSESN_Weekly[:, 101:bseTime])

bestParameters_BSESN_Weekly = collect(output__BSESN_Weekly[argmin(output__BSESN_Weekly.Objective), 1:4])

@save "Data/xu-calibration/prices-bsesn-weekly.jld2" priceMatrix_XU_BSESN_Weekly
@save "Data/xu-calibration/log-returns-bsesn-weekly.jld2" logReturnsMatrix_XU_BSESN_Weekly
@save "Data/xu-calibration/parameters-bsesn-weekly.jld2" bestParameters_BSESN_Weekly
@save "Data/xu-calibration/objective-results-bsesn-weekly.jld2" output__BSESN_Weekly
@save "Data/xu-calibration/xu-index-price-bsesn-weekly.jld2" xuIndexPrice_BSESN_Weekly
@save "Data/xu-calibration/xu-index-log-returns-bsesn-weekly.jld2" xuIndexReturn_BSESN_Weekly

#####################################################################