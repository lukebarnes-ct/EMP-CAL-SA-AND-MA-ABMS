
##### FW Model Sensitivity Analysis

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

# Franke and Westerhoff ABM

function fwABM(Time, n, beta, chi, phi, sigma_C, sigma_F, alpha_0, alpha_N, alpha_P)

    ### Parameters

    mu = 0.01

    # Set Seed for Reproducibility
    Random.seed!(n)

    # Number of Timesteps
    T = Time

    # Fundamental Value
    fundValue = zeros(T)
    fundValue[1] = 10

    ### Initialise Variables and Matrices
    
    # Prices of the Risky Asset
    price = zeros(T)
    price[1] = 10

    # Total Returns of Risky Assets
    returns = zeros(T)

    # Log Returns of Risky Assets
    log_returns = zeros(T)

    # Fundamentalists Demand of the Risky Asset
    demand_Fund = zeros(T)

    # Chartists Demand of the Risky Asset
    demand_Chart = zeros(T)

    # Relative Fitness

    relFit = zeros(T)

    # Percentage of Fundamentalists
    n_Fund = zeros(T)

    # Percentage of Chartists
    n_Chart = zeros(T)

    for t in 2:T

        price[t] = price[t-1] + mu * (n_Chart[t-1] * demand_Chart[t-1] + n_Fund[t-1] * demand_Fund[t-1])

        fundValue[t] =  10

        # Chartists Demand of the Risky Asset at time t

        error_C = rand(Normal(0, sigma_C), 1)[1]

        demand_Chart[t] = chi * (price[t] - price[t-1]) + error_C

        # Fundamentalists Demand of the Risky Asset at time t

        error_F = rand(Normal(0, sigma_F), 1)[1]

        demand_Fund[t] = phi * (fundValue[t] - price[t]) + error_F

        # Chartists Share of the Risky Asset market at time t
        n_Chart[t] = 1/(1 + exp(-beta * relFit[t-1]))

        # Fundamentalists Share of the Risky Asset market at time t
        n_Fund[t] = 1 - n_Chart[t]

        # Relative Fitness at time t
        relFit[t] = alpha_0 + alpha_N * (n_Fund[t] - n_Chart[t]) + alpha_P * (price[t] - fundValue[t])^2

        returns[t] = ((price[t] - price[t-1]) / price[t-1]) * 20
        
        # Log Return at time t
        safeLog(x) = x > 0 ? log(x) : log_returns[t-3]
        log_returns[t] = safeLog(price[t] / price[t-1])
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

    BETA = par[1]
    CHI = par[2]
    PHI = par[3]
    SIGMA_C = par[4]
    SIGMA_F = par[5]
    ALPHA_0 = par[6]
    ALPHA_N = par[7]
    ALPHA_P = par[8]

    for n in 1:N

        prices, returns = fwABM(timeEnd, n, BETA, CHI, PHI, SIGMA_C, SIGMA_F, ALPHA_0, ALPHA_N, ALPHA_P)

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

function f_FW(x, repetitions, index, timescale)

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

@load "Data/fw-calibration/parameters-jse-daily.jld2" optParam_FW_JSE_Daily

optVal_FW_JSE_Daily = f_FW(optParam_FW_JSE_Daily, 10, "JSE", "Daily")

function fwSensitivityAnalysis(index, timeframe, combo, params)

    beta_range = collect(0:0.0025:0.1)
    chi_range = collect(0:0.02:1)
    phi_range = collect(0:0.5:15)
    sigma_c_range = collect(0.5:0.5:15)
    sigma_f_range = collect(0:0.5:10)
    alpha_0_range = collect(-1000:100:1000)
    alpha_n_range = collect(0.01:5:100)
    alpha_p__range = collect(0.01:5:100)

    fwResults = DataFrame(beta = Float64[], 
                          chi = Float64[], 
                          phi = Float64[], 
                          sigma_c = Float64[], 
                          sigma_f = Float64[], 
                          alpha_0 = Float64[], 
                          alpha_n = Float64[], 
                          alpha_p = Float64[],
                          value = Float64[])

    if combo == "beta_chi"

        for beta in beta_range

            println(beta)

            for chi in chi_range

                optVal = f_FW([beta, chi, params[3], params[4], params[5], params[6], params[7], params[8]], 
                              10, index, timeframe)
                push!(fwResults, (beta, chi, params[3], params[4], params[5], params[6], params[7], params[8], optVal))
            end
        end

    elseif combo == "phi_sigma_c"

        for phi in phi_range

            println(phi)

            for sigma_c in sigma_c_range

                optVal = f_FW([params[1], params[2], phi, sigma_c, params[5], params[6], params[7], params[8]], 
                              10, index, timeframe)
                push!(fwResults, (params[1], params[2], phi, sigma_c, params[5], params[6], params[7], params[8], optVal))
            end
        end

    elseif combo == "sigma_f_alpha_0"

        for sigma_f in sigma_f_range

            println(sigma_f)

            for alpha_0 in alpha_0_range

                optVal = f_FW([params[1], params[2], params[3], params[4], sigma_f, alpha_0, params[7], params[8]], 
                              10, index, timeframe)
                push!(fwResults, (params[1], params[2], params[3], params[4], sigma_f, alpha_0, params[7], params[8], optVal))
            end
        end
    
    elseif combo == "alpha_n_alpha_p"

        for alpha_n in alpha_n_range

            println(alpha_n)

            for alpha_p in alpha_p_range

                optVal = f_FW([params[1], params[2], params[3], params[4], params[5], params[6], alpha_n, alpha_p], 
                              10, index, timeframe)
                push!(fwResults, (params[1], params[2], params[3], params[4], params[5], params[6], alpha_n, alpha_p, optVal))
            end
        end
    end

    return fwResults
end

#####################################################################

results_beta_chi = fwSensitivityAnalysis("JSE", "Daily", "beta_chi", optParam_FW_JSE_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "β", ylabel = "χ", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_beta_chi.beta, 
                        results_beta_chi.chi, results_beta_chi.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_FW_JSE_Daily[1]], [optParam_FW_JSE_Daily[2]], [optVal_FW_JSE_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/fw-calibration/sensitivity-analysis/jse_daily_beta_chi.png", 
     fig, px_per_unit = 4)

#####################################################################

results_phi_sigma_c = fwSensitivityAnalysis("JSE", "Daily", "phi_sigma_c", optParam_FW_JSE_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "ϕ", ylabel = L"σ^2_C", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_phi_sigma_c.phi, 
                        results_phi_sigma_c.sigma_c, results_phi_sigma_c.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_FW_JSE_Daily[3]], [optParam_FW_JSE_Daily[4]], [optVal_FW_JSE_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/fw-calibration/sensitivity-analysis/jse_daily_phi_sigma_c.png", 
     fig, px_per_unit = 4)

#####################################################################

results_sigma_f_alpha_0 = fwSensitivityAnalysis("JSE", "Daily", "sigma_f_alpha_0", optParam_FW_JSE_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "β", ylabel = "χ", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_sigma_f_alpha_0.sigma_f, 
                        results_sigma_f_alpha_0.alpha_0, results_sigma_f_alpha_0.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_FW_JSE_Daily[5]], [optParam_FW_JSE_Daily[6]], [optVal_FW_JSE_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/fw-calibration/sensitivity-analysis/jse_daily_sigma_f_alpha_0.png", 
     fig, px_per_unit = 4)

#####################################################################

results_alpha_n_alpha_p = fwSensitivityAnalysis("JSE", "Daily", "alpha_n_alpha_p", optParam_FW_JSE_Daily)

fig = Figure()
axis = Axis3(fig[1, 1], xlabel = "β", ylabel = "χ", zlabel = "Objective Value")
surf = GLMakie.surface!(axis, results_alpha_n_alpha_p.alpha_n, 
                        results_alpha_n_alpha_p.alpha_p, results_alpha_n_alpha_p.value, 
                        colormap =:plasma, label = "f(0)")
Colorbar(fig[1, 2], surf)
GLMakie.scatter!(axis, [optParam_FW_JSE_Daily[7]], [optParam_FW_JSE_Daily[8]], [optVal_FW_JSE_Daily + 5], 
                 color =:red, markersize = 10, strokewidth = 2, strokecolor =:black)
fig

save("Plots/fw-calibration/sensitivity-analysis/jse_daily_alpha_n_alpha_p.png", 
     fig, px_per_unit = 4)

