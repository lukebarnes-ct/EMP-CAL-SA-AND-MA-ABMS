##### Model Calibration (Method of Simulated Moments)

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

    # Dividends of the Risky Asset
    dividends = zeros(T)
    
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

        # fundValue[t] =  (1 + phi + (phiSD * delta)) * fundValue[t-1]
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
        log_returns[t] = log(price[t]/price[t-1])
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

    simMom = getSimulatedMoments(x, repetitions, "F&W", index, timescale)

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

f_FW_MBB(x, grad) = f_FW(x, repetitions, index, timescale)

#####################################################################

function FW_nelderMeadSimulation(threshold)
    
    perturb = 0.25

    lowerBounds = [0, 0, 0, 0, 0, -1000, 0.01, 0.01]
    upperBounds = [0.1, 1, 10, 10, 10, 1000, 100, 100]

    initialParameters = [rand() * (x - l) + l for (l, x) in zip(lowerBounds, upperBounds)]

    opt = Opt(:LN_NELDERMEAD, length(initialParameters))
    opt.xtol_rel = 1e-6
    opt.lower_bounds = lowerBounds
    opt.upper_bounds = upperBounds
    opt.min_objective = f_FW_MBB

    (currentValue, currentParameters, ret) = NLopt.optimize(opt, initialParameters)

    bestParameters = currentParameters
    bestValue = currentValue

    println("Best Value is: ", bestValue)

    minThreshold = opt.xtol_rel

    counter = 1

    while threshold > minThreshold

        println("Counter: $counter")

        newParameters = [x * (1 + rand(MersenneTwister(counter), Uniform(-perturb, perturb))) for x in bestParameters]
        newParameters = [clamp(x, lower, upper) for (x, lower, upper) in zip(newParameters, lowerBounds, upperBounds)]

        optCurrent = Opt(:LN_NELDERMEAD, length(newParameters))
        optCurrent.xtol_rel = 1e-6
        optCurrent.lower_bounds = lowerBounds
        optCurrent.upper_bounds = upperBounds
        optCurrent.min_objective = f_FW_MBB

        (currentValue, currentParameters, ret) = NLopt.optimize(optCurrent, newParameters)
            
        println(currentValue)

        if (currentValue < bestValue) || (currentValue - bestValue < threshold)

            println("NEW BEST PARAMETERS: $currentParameters")
            println("NEW BEST VALUE: $currentValue")

            bestValue = currentValue
            bestParameters = currentParameters
        end

        threshold = threshold * 0.95
        counter = counter + 1
    end

    return bestParameters, bestValue
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

# Run the MSM Nelder Mead Optimisation for each Empirical Log Return Time Series

tol_FW = 0.00001

repetitions = 10

# JSE

index = "JSE"

# JSE Daily Log Returns 

timescale = "Daily"

optParam_FW_JSE_Daily, optValue_FW_JSE_Daily = FW_nelderMeadSimulation(tol_FW)
prices_FW_JSE_Daily, logReturns_FW_JSE_Daily = fwABM(modelSims, id,
                                                     optParam_FW_JSE_Daily[1], optParam_FW_JSE_Daily[2], 
                                                     optParam_FW_JSE_Daily[3], optParam_FW_JSE_Daily[4], 
                                                     optParam_FW_JSE_Daily[5], optParam_FW_JSE_Daily[6], 
                                                     optParam_FW_JSE_Daily[7], optParam_FW_JSE_Daily[8])

@save "Data/fw-calibration/prices-jse-daily.jld2" prices_FW_JSE_Daily
@save "Data/fw-calibration/log-returns-jse-daily.jld2" logReturns_FW_JSE_Daily

# JSE Weekly Log Returns 

timescale = "Weekly"

optParam_FW_JSE_Weekly, optValue_FW_JSE_Weekly = FW_nelderMeadSimulation(tol_FW)
prices_FW_JSE_Weekly, logReturns_FW_JSE_Weekly = fwABM(modelSims, id,
                                                       optParam_FW_JSE_Weekly[1], optParam_FW_JSE_Weekly[2], 
                                                       optParam_FW_JSE_Weekly[3], optParam_FW_JSE_Weekly[4], 
                                                       optParam_FW_JSE_Weekly[5], optParam_FW_JSE_Weekly[6], 
                                                       optParam_FW_JSE_Weekly[7], optParam_FW_JSE_Weekly[8])

@save "Data/fw-calibration/prices-jse-weekly.jld2" prices_FW_JSE_Weekly
@save "Data/fw-calibration/log-returns-jse-weekly.jld2" logReturns_FW_JSE_Weekly

# SSE50

index = "SSE"

# SSE50 Daily Log Returns 

timescale = "Daily"

optParam_FW_SSE50_Daily, optValue_FW_SSE50_Daily = FW_nelderMeadSimulation(tol_FW)
prices_FW_SSE50_Daily, logReturns_FW_SSE50_Daily = fwABM(modelSims, id,
                                                         optParam_FW_SSE50_Daily[1], optParam_FW_SSE50_Daily[2], 
                                                         optParam_FW_SSE50_Daily[3], optParam_FW_SSE50_Daily[4], 
                                                         optParam_FW_SSE50_Daily[5], optParam_FW_SSE50_Daily[6], 
                                                         optParam_FW_SSE50_Daily[7], optParam_FW_SSE50_Daily[8])

@save "Data/fw-calibration/prices-sse50-daily.jld2" prices_FW_SSE50_Daily
@save "Data/fw-calibration/log-returns-sse50-daily.jld2" logReturns_FW_SSE50_Daily

# SSE50 Weekly Log Returns 

timescale = "Weekly"

optParam_FW_SSE50_Weekly, optValue_FW_SSE50_Weekly = FW_nelderMeadSimulation(tol_FW)
prices_FW_SSE50_Weekly, logReturns_FW_SSE50_Weekly = fwABM(modelSims, id,
                                                           optParam_FW_SSE50_Weekly[1], optParam_FW_SSE50_Weekly[2], 
                                                           optParam_FW_SSE50_Weekly[3], optParam_FW_SSE50_Weekly[4], 
                                                           optParam_FW_SSE50_Weekly[5], optParam_FW_SSE50_Weekly[6], 
                                                           optParam_FW_SSE50_Weekly[7], optParam_FW_SSE50_Weekly[8])

@save "Data/fw-calibration/prices-sse50-weekly.jld2" prices_FW_SSE50_Weekly
@save "Data/fw-calibration/log-returns-sse50-weekly.jld2" logReturns_FW_SSE50_Weekly

# BSESN

index = "BSE"

# BSESN Daily Log Returns 

timescale = "Daily"

optParam_FW_BSESN_Daily, optValue_FW_BSESN_Daily = FW_nelderMeadSimulation(tol_FW)
prices_FW_BSESN_Daily, logReturns_FW_BSESN_Daily = fwABM(modelSims, id,
                                                         optParam_FW_BSESN_Daily[1], optParam_FW_BSESN_Daily[2], 
                                                         optParam_FW_BSESN_Daily[3], optParam_FW_BSESN_Daily[4], 
                                                         optParam_FW_BSESN_Daily[5], optParam_FW_BSESN_Daily[6], 
                                                         optParam_FW_BSESN_Daily[7], optParam_FW_BSESN_Daily[8])

@save "Data/fw-calibration/prices-bsesn-daily.jld2" prices_FW_BSESN_Daily
@save "Data/fw-calibration/log-returns-bsesn-daily.jld2" logReturns_FW_BSESN_Daily

# BSESN Weekly Log Returns 

timescale = "Weekly"

optParam_FW_BSESN_Weekly, optValue_FW_BSESN_Weekly = FW_nelderMeadSimulation(tol_FW)
prices_FW_BSESN_Weekly, logReturns_FW_BSESN_Weekly = fwABM(modelSims, id,
                                                           optParam_FW_BSESN_Weekly[1], optParam_FW_BSESN_Weekly[2], 
                                                           optParam_FW_BSESN_Weekly[3], optParam_FW_BSESN_Weekly[4], 
                                                           optParam_FW_BSESN_Weekly[5], optParam_FW_BSESN_Weekly[6], 
                                                           optParam_FW_BSESN_Weekly[7], optParam_FW_BSESN_Weekly[8])

@save "Data/fw-calibration/prices-bsesn-weekly.jld2" prices_FW_BSESN_Weekly
@save "Data/fw-calibration/log-returns-bsesn-weekly.jld2" logReturns_FW_BSESN_Weekly

#####################################################################