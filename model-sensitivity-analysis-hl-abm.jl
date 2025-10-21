
##### HL Model Sensitivity Analysis

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
using GLMakie
using LaTeXStrings

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

# He and Li ABM

function hlABM(Time, n,  n1, n2, mu, gamma, delta, alpha)

    ### Parameters

    # Set Seed for Reproducibility
    Random.seed!(n)

    # Number of Timesteps
    T = Time

    # Risk Free Rate
    r = 0.05

    # Gross Return
    K = 250
    R = 1 + (r/K)

    # Initial Fundamental Value
    in_Fund = 100

    # Initial Dividend
    in_Div = in_Fund * (R - 1)

    # Constant Variance
    sigma_C = ((in_Fund * 0.2)^2)/K

    # Dividend Variance
    sigma_Div = 1/K

    # Fundamental Value Variance
    sigma_Epsilon = (20 / sqrt(K))/100

    # Noisy Demand Variance
    sigma_Zeta = 1

    # Chartist Variance Influence
    b2 = 1
    # b = b2/sigma_C
    b = 1

    # Demand Parameters
    a1 = 0.8
    a2 = 0.8

    q = r^2
    m = n1 - n2

    ### Initialise Variables and Matrices

    # Fundamental Value
    fundValue = zeros(T)

    # Dividends of the Risky Asset
    dividends = zeros(T)
    
    # Prices of the Risky Asset
    price = zeros(T)

    # Returns of the Risky Asset
    returns = zeros(T)

    # Returns of the Risky Asset
    log_returns = zeros(T)

    # Excess Capital Gain
    excGain = zeros(T)

    # Noisy Demand
    zeta = zeros(T)

    # Sample Mean and Variance
    u = zeros(T)
    v = zeros(T)

    # Fundamentalists Wealth
    wealth_Fund = zeros(T)

    # Chartists Wealth
    wealth_Chart = zeros(T)

    # Fundamentalists Expected Return of the Risky Asset
    expRet_Fund = zeros(T)

    # Chartists Expected Return of the Risky Asset
    expRet_Chart = zeros(T)

    # Fundamentalists Expected Variance of the Risky Asset
    expVar_Fund = zeros(T)

    # Chartists Expected Variance of the Risky Asset
    expVar_Chart = zeros(T)

    # Fundamentalists Demand of the Risky Asset
    demand_Fund = zeros(T)

    # Chartists Demand of the Risky Asset
    demand_Chart = zeros(T)

    # T = 1
    price[1] = in_Fund
    fundValue[1] = in_Fund
    demand_Chart[1] = 0
    demand_Fund[1] = 0
    wealth_Chart[1] = 100
    wealth_Fund[1] = 100
    u[1] = (1 - delta) * price[1]
    v[1] = delta * (1 - delta) * (price[1] - u[1])^2

    for t in 2:T

        # Epsilon Error Term
        zeta[t] = rand(Normal(0, sigma_Zeta), 1)[1]

        # Price of the risky asset at time t
        price[t] = price[t-1] + ((mu/2) * (((1 + m) * (demand_Fund[t-1])) + ((1 - m) * (demand_Chart[t-1])))) + zeta[t]
        
        # Epsilon Error Term
        epsilon = rand(Normal(0, 1), 1)[1]

        # Fundamental Value at time t
        fundValue[t] = fundValue[t-1] * (1 + (sigma_Epsilon * epsilon))

        # Dividend at time t
        dividends[t] = rand(Normal(in_Div, sigma_Div), 1)[1]

        # Risky Asset Return at time t
        returns[t] = (price[t] + dividends[t] - price[t-1]) / price[t-1]

        # Log Return at time t
        safeLog(x) = x > 0 ? log(x) : price[t-3]
        log_returns[t] = safeLog(price[t] / price[t-1])

        # Excess Capital Gain at time t
        excGain[t] = price[t] + dividends[t] - (R * price[t-1])

        # Fundamentalist Wealth at time t 
        wealth_Fund[t] = (R * wealth_Fund[t-1]) + (excGain[t] * demand_Fund[t-1])

        # Chartist Wealth at time t 
        wealth_Chart[t] = (R * wealth_Chart[t-1]) + (excGain[t] * demand_Chart[t-1])

        # Fundamentalists Expected Return at time t + 1
        expRet_Fund[t] = (alpha * (fundValue[t] - price[t])) - ((R - 1) * (price[t] - in_Fund))

        # Fundamentalists Expected Variance at time t + 1
        expVar_Fund[t] = (1 + q) * sigma_C

        # Fundamentalists Demand at time t
        demand_Fund[t] = (expRet_Fund[t]) / (a1 * expVar_Fund[t])

        # Sample Mean at time t
        u[t] = (delta * u[t-1]) + ((1 - delta) * price[t])

        # Sample Variance at time t
        v[t] = (delta * v[t-1]) + (delta * (1 - delta) * (price[t] - u[t-1])^2)

        # Chartists Expected Return at time t + 1
        expRet_Chart[t] = (gamma * (price[t] - u[t])) - ((R - 1) * (price[t] - in_Fund))

        # Chartists Expected Variance at time t + 1
        expVar_Chart[t] = sigma_C * (1 + q + (b * v[t]))

        # Chartists Demand at time t
        demand_Chart[t] = (expRet_Chart[t]) / (a2 * expVar_Chart[t])
        
    end

    return price, log_returns

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
    
    timeBegin = 1
    timeEnd = 10000

    sMoments = zeros(6, N)

    MU = par[1]
    GAMMA = par[2]
    DELTA = par[3] 
    ALPHA = par[4]

    for n in 1:N

        prices, returns = hlABM(timeEnd, n, 20, 20, MU, GAMMA, DELTA, ALPHA)

        moments = getMoments(returns, timeBegin, timeEnd, "Simulated", index, timescale)

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

function f_HL(x, repetitions, index, timescale)

    simMom = getSimulatedMoments(x, repetitions, index, timescale)

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

# Sensitivity Analysis

@load "Data/hl-calibration/parameters-sse50-daily.jld2" optParam_HL_SSE50_Daily

optVal_HL_SSE50_Daily = f_HL(optParam_HL_SSE50_Daily, 10, "SSE", "Daily")

function hlSensitivityAnalysis(index, timeframe, combo, params)

    mu_range = collect(0.5:0.5:15)
    gamma_range = collect(0:0.5:8)
    delta_range = collect(0.01:0.02:0.99)
    alpha_range = collect(0.2:0.02:0.5)

    hlResults = DataFrame(mu = Float64[], 
                          gamma = Float64[], 
                          delta = Float64[], 
                          alpha = Float64[], 
                          value = Float64[])

    if combo == "mu_gamma"

        for mu in mu_range

            println(mu)
            for gamma in gamma_range

                optVal = f_HL([mu, gamma, params[3], params[4]], 10, index, timeframe)
                push!(hlResults, (mu, gamma, params[3], params[4], optVal))
            end
        end

    elseif combo == "mu_delta"

        for mu in mu_range

            for delta in delta_range

                optVal = f_HL([mu, params[2], delta, params[4]], 10, index, timeframe)
                push!(hlResults, (mu, params[2], delta, params[4], optVal))
            end
        end

    elseif combo == "mu_alpha"

        for mu in mu_range

            for alpha in alpha_range

                optVal = f_HL([mu, params[2], params[3], alpha], 10, index, timeframe)
                push!(hlResults, (mu, params[2], params[3], alpha, optVal))
            end
        end
    
    elseif combo == "delta_alpha"

        for delta in delta_range

            println(delta)
            for alpha in alpha_range

                optVal = f_HL([params[1], params[2], delta, alpha], 10, index, timeframe)
                push!(hlResults, (params[1], params[2], delta, alpha, optVal))
            end
        end
    end

    return hlResults
end

#####################################################################

results_mu_gamma = hlSensitivityAnalysis("SSE", "Daily", "mu_gamma", optParam_HL_SSE50_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "μ", ylabel = "γ", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_mu_gamma.mu, 
                        results_mu_gamma.gamma, results_mu_gamma.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_HL_SSE50_Daily[1]], [optParam_HL_SSE50_Daily[2]], [optVal_HL_SSE50_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/hl-calibration/sensitivity-analysis/sse50_daily_mu_gamma.png", 
     fig, px_per_unit = 4)

#####################################################################

results_delta_alpha = hlSensitivityAnalysis("SSE", "Daily", "delta_alpha", optParam_HL_SSE50_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "δ", ylabel = "α", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_delta_alpha.delta, 
                        results_delta_alpha.alpha, results_delta_alpha.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_HL_SSE50_Daily[3]], [optParam_HL_SSE50_Daily[4]], [optVal_HL_SSE50_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/hl-calibration/sensitivity-analysis/sse50_daily_delta_alpha.png", 
     fig, px_per_unit = 4)


