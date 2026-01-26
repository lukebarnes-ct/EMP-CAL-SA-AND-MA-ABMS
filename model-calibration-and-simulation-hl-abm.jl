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

f_HL_MBB(x, grad) = f_HL(x, repetitions, index, timescale)

#####################################################################

function HL_nelderMeadSimulation(threshold)
    
    perturb = 0.50

    lowerBounds = [0, 0, 0, 0]
    upperBounds = [10, 10, 1, 1]

    initialParameters = [rand() * (x - l) + l for (l, x) in zip(lowerBounds, upperBounds)]

    opt = Opt(:LN_NELDERMEAD, length(initialParameters))
    opt.xtol_rel = 1e-6
    opt.lower_bounds = lowerBounds
    opt.upper_bounds = upperBounds
    opt.min_objective = f_HL_MBB

    (currentValue, currentParameters, ret) = NLopt.optimize(opt, initialParameters)

    bestParameters = currentParameters
    bestValue = currentValue

    minThreshold = opt.xtol_rel

    counter = 1

    while threshold > minThreshold

        println("Counter: $counter")

        newParameters = [x * (1 + rand(MersenneTwister(counter), Uniform(-perturb, perturb))) for x in bestParameters]
        newParameters = [clamp(x, lower, upper) for (x, lower, upper) in zip(newParameters, lowerBounds, upperBounds)]

        optCurrent = Opt(:LN_NELDERMEAD, length(newParameters))
        maxtime!(optCurrent, 5.0)
        optCurrent.xtol_rel = 1e-6
        optCurrent.lower_bounds = lowerBounds
        optCurrent.upper_bounds = upperBounds
        optCurrent.min_objective = f_HL_MBB

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

tol_HL = 10

repetitions = 10
modelSims = 10000
id = 1

# JSE

index = "JSE"

# JSE Daily Log Returns 

timescale = "Daily"

optParam_HL_JSE_Daily, optValue_HL_JSE_Daily = HL_nelderMeadSimulation(tol_HL)
prices_HL_JSE_Daily, logReturns_HL_JSE_Daily = hlABM(modelSims, id, 20, 20, 
                                                     optParam_HL_JSE_Daily[1], optParam_HL_JSE_Daily[2], 
                                                     optParam_HL_JSE_Daily[3], optParam_HL_JSE_Daily[4])

@save "Data/hl-calibration/prices-jse-daily.jld2" prices_HL_JSE_Daily
@save "Data/hl-calibration/log-returns-jse-daily.jld2" logReturns_HL_JSE_Daily
@save "Data/hl-calibration/parameters-jse-daily.jld2" optParam_HL_JSE_Daily

# JSE Weekly Log Returns 

timescale = "Weekly"

optParam_HL_JSE_Weekly, optValue_HL_JSE_Weekly = HL_nelderMeadSimulation(tol_HL)
prices_HL_JSE_Weekly, logReturns_HL_JSE_Weekly = hlABM(modelSims, id, 20, 20, 
                                                       optParam_HL_JSE_Weekly[1], optParam_HL_JSE_Weekly[2], 
                                                       optParam_HL_JSE_Weekly[3], optParam_HL_JSE_Weekly[4])

@save "Data/hl-calibration/prices-jse-weekly.jld2" prices_HL_JSE_Weekly
@save "Data/hl-calibration/log-returns-jse-weekly.jld2" logReturns_HL_JSE_Weekly
@save "Data/hl-calibration/parameters-jse-weekly.jld2" optParam_HL_JSE_Weekly

# SSE50

index = "SSE"

# SSE50 Daily Log Returns 

timescale = "Daily"

optParam_HL_SSE50_Daily, optValue_HL_SSE50_Daily = HL_nelderMeadSimulation(tol_HL)
prices_HL_SSE50_Daily, logReturns_HL_SSE50_Daily = hlABM(modelSims, id, 20, 20, 
                                                         optParam_HL_SSE50_Daily[1], optParam_HL_SSE50_Daily[2], 
                                                         optParam_HL_SSE50_Daily[3], optParam_HL_SSE50_Daily[4])

@save "Data/hl-calibration/prices-sse50-daily.jld2" prices_HL_SSE50_Daily
@save "Data/hl-calibration/log-returns-sse50-daily.jld2" logReturns_HL_SSE50_Daily
@save "Data/hl-calibration/parameters-sse50-daily.jld2" optParam_HL_SSE50_Daily

# SSE50 Weekly Log Returns 

timescale = "Weekly"

optParam_HL_SSE50_Weekly, optValue_HL_SSE50_Weekly = HL_nelderMeadSimulation(tol_HL)
prices_HL_SSE50_Weekly, logReturns_HL_SSE50_Weekly = hlABM(modelSims, id, 20, 20, 
                                                           optParam_HL_SSE50_Weekly[1], optParam_HL_SSE50_Weekly[2], 
                                                           optParam_HL_SSE50_Weekly[3], optParam_HL_SSE50_Weekly[4])

@save "Data/hl-calibration/prices-sse50-weekly.jld2" prices_HL_SSE50_Weekly
@save "Data/hl-calibration/log-returns-sse50-weekly.jld2" logReturns_HL_SSE50_Weekly
@save "Data/hl-calibration/parameters-sse50-weekly.jld2" optParam_HL_SSE50_Weekly

# BSESN

index = "BSE"

# BSESN Daily Log Returns 

timescale = "Daily"

optParam_HL_BSESN_Daily, optValue_HL_BSESN_Daily = HL_nelderMeadSimulation(tol_HL)
prices_HL_BSESN_Daily, logReturns_HL_BSESN_Daily = hlABM(modelSims, id, 20, 20, 
                                                         optParam_HL_BSESN_Daily[1], optParam_HL_BSESN_Daily[2], 
                                                         optParam_HL_BSESN_Daily[3], optParam_HL_BSESN_Daily[4])

@save "Data/hl-calibration/prices-bsesn-daily.jld2" prices_HL_BSESN_Daily
@save "Data/hl-calibration/log-returns-bsesn-daily.jld2" logReturns_HL_BSESN_Daily
@save "Data/hl-calibration/parameters-bsesn-daily.jld2" optParam_HL_BSESN_Daily

# BSESN Weekly Log Returns 

timescale = "Weekly"

optParam_HL_BSESN_Weekly, optValue_HL_BSESN_Weekly = HL_nelderMeadSimulation(tol_HL)
prices_HL_BSESN_Weekly, logReturns_HL_BSESN_Weekly = hlABM(modelSims, id, 20, 20, 
                                                           optParam_HL_BSESN_Weekly[1], optParam_HL_BSESN_Weekly[2], 
                                                           optParam_HL_BSESN_Weekly[3], optParam_HL_BSESN_Weekly[4])

@save "Data/hl-calibration/prices-bsesn-weekly.jld2" prices_HL_BSESN_Weekly
@save "Data/hl-calibration/log-returns-bsesn-weekly.jld2" logReturns_HL_BSESN_Weekly
@save "Data/hl-calibration/parameters-bsesn-weekly.jld2" optParam_HL_BSESN_Weekly

#####################################################################

# Moment Confidence Intervals

using LinearAlgebra
using DataFrames
using PrettyTables

function getConfidenceInterval(moments, InvBootMatrix)

    sd = sqrt.(diag(inv(InvBootMatrix)))
    band = 1.96 .* sd
    lower = moments .- band
    upper = moments .+ band

    return lower, upper
end

function empiricalConfidenceInterval(empMom, empMBBM)

    lowerCI, upperCI = getConfidenceInterval(empMom, empMBBM)

    return lowerCI, upperCI
end

function calibratedConfidenceInterval(id, tEnd, calRet, index, timescale)

    calMom = getMoments(calRet, 1001, tEnd, "Simulated", index, timescale)
    
    blockWindow = 100
    blockSamples = 1000
    calMBBM = getMovingBlockBootstrapMatrix(id, calRet, blockWindow, blockSamples)

    lowerCI, upperCI = getConfidenceInterval(calMom, calMBBM)

    return lowerCI, upperCI, calMom
end

function getSimulatedMomentsAndReturns(par, N, index, timescale)
    
    timeBegin = 1
    timeEnd = 10000

    sMoments = zeros(6, N)
    sReturns = zeros(timeEnd, N)

    MU = par[1]
    GAMMA = par[2]
    DELTA = par[3] 
    ALPHA = par[4]

    for n in 1:N

        prices, returns = hlABM(timeEnd, n, 20, 20, MU, GAMMA, DELTA, ALPHA)

        moments = getMoments(returns, timeBegin, timeEnd, "Simulated", index, timescale)

        sMoments[:, n] = moments
        sReturns[:, n] = returns
    end

    simMoments = mean(sMoments, dims = 2)

    simReturns = mean(sReturns, dims = 2)

    return simMoments, simReturns

end

function simulatedConfidenceInterval(id, par, reps, index, timescale)

    simMom, simRet = getSimulatedMomentsAndReturns(par, reps, index, timescale)

    blockWindow = 100
    blockSamples = 1000
    simMBBM = getMovingBlockBootstrapMatrix(id, simRet, blockWindow, blockSamples)

    lowerCI, upperCI = getConfidenceInterval(simMom, simMBBM)

    return lowerCI, upperCI
end

#####################################################################

# Empirical Confidence Intervals

emp_lowerCI_JSE_Daily, emp_upperCI_JSE_Daily = empiricalConfidenceInterval(momentsJSE_Daily, bootstrapMatrixJSE_Daily)

latexRow_JSE_Daily = vec(hcat(round.(emp_lowerCI_JSE_Daily, digits = 4), momentsJSE_Daily, round.(emp_upperCI_JSE_Daily, digits = 4))')
latex_df_JSE_Daily = DataFrame(latexRow_JSE_Daily', :auto)
latex_output_JSE_Daily = pretty_table(latex_df_JSE_Daily, backend = :latex)

emp_lowerCI_JSE_Weekly, emp_upperCI_JSE_Weekly = empiricalConfidenceInterval(momentsJSE_Weekly, bootstrapMatrixJSE_Weekly)

latexRow_JSE_Weekly = vec(hcat(round.(emp_lowerCI_JSE_Weekly, digits = 4), momentsJSE_Weekly, round.(emp_upperCI_JSE_Weekly, digits = 4))')
latex_df_JSE_Weekly = DataFrame(latexRow_JSE_Weekly', :auto)
latex_output_JSE_Weekly = pretty_table(latex_df_JSE_Weekly, backend = :latex)

emp_lowerCI_SSE50_Daily, emp_upperCI_SSE50_Daily = empiricalConfidenceInterval(momentsSSE50_Daily, bootstrapMatrixSSE50_Daily)

latexRow_SSE50_Daily = vec(hcat(round.(emp_lowerCI_SSE50_Daily, digits = 4), momentsSSE50_Daily, round.(emp_upperCI_SSE50_Daily, digits = 4))')
latex_df_SSE50_Daily = DataFrame(latexRow_SSE50_Daily', :auto)
latex_output_SSE50_Daily = pretty_table(latex_df_SSE50_Daily, backend = :latex)

emp_lowerCI_SSE50_Weekly, emp_upperCI_SSE50_Weekly = empiricalConfidenceInterval(momentsSSE50_Weekly, bootstrapMatrixSSE50_Weekly)

latexRow_SSE50_Weekly = vec(hcat(round.(emp_lowerCI_SSE50_Weekly, digits = 4), momentsSSE50_Weekly, round.(emp_upperCI_SSE50_Weekly, digits = 4))')
latex_df_SSE50_Weekly = DataFrame(latexRow_SSE50_Weekly', :auto)
latex_output_SSE50_Weekly = pretty_table(latex_df_SSE50_Weekly, backend = :latex)

emp_lowerCI_BSESN_Daily, emp_upperCI_BSESN_Daily = empiricalConfidenceInterval(momentsBSESN_Daily, bootstrapMatrixBSESN_Daily)

latexRow_BSESN_Daily = vec(hcat(round.(emp_lowerCI_BSESN_Daily, digits = 4), momentsBSESN_Daily, round.(emp_upperCI_BSESN_Daily, digits = 4))')
latex_df_BSESN_Daily = DataFrame(latexRow_BSESN_Daily', :auto)
latex_output_BSESN_Daily = pretty_table(latex_df_BSESN_Daily, backend = :latex)

emp_lowerCI_BSESN_Weekly, emp_upperCI_BSESN_Weekly = empiricalConfidenceInterval(momentsBSESN_Weekly, bootstrapMatrixBSESN_Weekly)

latexRow_BSESN_Weekly = vec(hcat(round.(emp_lowerCI_BSESN_Weekly, digits = 4), momentsBSESN_Weekly, round.(emp_upperCI_BSESN_Weekly, digits = 4))')
latex_df_BSESN_Weekly = DataFrame(latexRow_BSESN_Weekly', :auto)
latex_output_BSESN_Weekly = pretty_table(latex_df_BSESN_Weekly, backend = :latex)

#####################################################################

plotStart_Daily = 1001
plotEnd_Daily_JSE = plotStart_Daily - 1 + lengthJSE_Daily 
plotEnd_Daily_SSE50 = plotStart_Daily - 1 + lengthSSE50_Daily
plotEnd_Daily_BSESN = plotStart_Daily - 1 + lengthBSESN_Daily

plotStart_Weekly = 1001
plotEnd_Weekly_JSE = plotStart_Weekly - 1 + lengthJSE_Weekly 
plotEnd_Weekly_SSE50 = plotStart_Weekly - 1 + lengthSSE50_Weekly
plotEnd_Weekly_BSESN = plotStart_Weekly - 1 + lengthBSESN_Weekly

#####################################################################

# Calibrated Confidence Intervals

cal_lowerCI_JSE_Daily, cal_upperCI_JSE_Daily, calMom_JSE_Daily = calibratedConfidenceInterval(id, plotEnd_Daily_JSE, logReturns_HL_JSE_Daily, "JSE", "Daily")

cal_latexRow_JSE_Daily = vec(hcat(round.(cal_lowerCI_JSE_Daily, digits = 4), calMom_JSE_Daily, round.(cal_upperCI_JSE_Daily, digits = 4))')
cal_latex_df_JSE_Daily = DataFrame(cal_latexRow_JSE_Daily', :auto)
cal_latex_output_JSE_Daily = pretty_table(cal_latex_df_JSE_Daily, backend = :latex)

cal_lowerCI_JSE_Weekly, cal_upperCI_JSE_Weekly, calMom_JSE_Weekly = calibratedConfidenceInterval(id, plotEnd_Weekly_JSE, logReturns_HL_JSE_Weekly, "JSE", "Weekly")

cal_latexRow_JSE_Weekly = vec(hcat(round.(cal_lowerCI_JSE_Weekly, digits = 4), calMom_JSE_Weekly, round.(cal_upperCI_JSE_Weekly, digits = 4))')
cal_latex_df_JSE_Weekly = DataFrame(cal_latexRow_JSE_Weekly', :auto)
cal_latex_output_JSE_Weekly = pretty_table(cal_latex_df_JSE_Weekly, backend = :latex)

cal_lowerCI_SSE50_Daily, cal_upperCI_SSE50_Daily, calMom_SSE50_Daily = calibratedConfidenceInterval(id, plotEnd_Daily_SSE50, logReturns_HL_SSE50_Daily, "SSE", "Daily")

cal_latexRow_SSE50_Daily = vec(hcat(round.(cal_lowerCI_SSE50_Daily, digits = 4), calMom_SSE50_Daily, round.(cal_upperCI_SSE50_Daily, digits = 4))')
cal_latex_df_SSE50_Daily = DataFrame(cal_latexRow_SSE50_Daily', :auto)
cal_latex_output_SSE50_Daily = pretty_table(cal_latex_df_SSE50_Daily, backend = :latex)

cal_lowerCI_SSE50_Weekly, cal_upperCI_SSE50_Weekly, calMom_SSE50_Weekly = calibratedConfidenceInterval(id, plotEnd_Weekly_SSE50, logReturns_HL_SSE50_Weekly, "SSE", "Weekly")

cal_latexRow_SSE50_Weekly = vec(hcat(round.(cal_lowerCI_SSE50_Weekly, digits = 4), calMom_SSE50_Weekly, round.(cal_upperCI_SSE50_Weekly, digits = 4))')
cal_latex_df_SSE50_Weekly = DataFrame(cal_latexRow_SSE50_Weekly', :auto)
cal_latex_output_SSE50_Weekly = pretty_table(cal_latex_df_SSE50_Weekly, backend = :latex)

cal_lowerCI_BSESN_Daily, cal_upperCI_BSESN_Daily, calMom_BSESN_Daily = calibratedConfidenceInterval(id, plotEnd_Daily_BSESN, logReturns_HL_BSESN_Daily, "BSE", "Daily")

cal_latexRow_BSESN_Daily = vec(hcat(round.(cal_lowerCI_BSESN_Daily, digits = 4), calMom_BSESN_Daily, round.(cal_upperCI_BSESN_Daily, digits = 4))')
cal_latex_df_BSESN_Daily = DataFrame(cal_latexRow_BSESN_Daily', :auto)
cal_latex_output_BSESN_Daily = pretty_table(cal_latex_df_BSESN_Daily, backend = :latex)

cal_lowerCI_BSESN_Weekly, cal_upperCI_BSESN_Weekly, calMom_BSESN_Weekly = calibratedConfidenceInterval(id, plotEnd_Weekly_BSESN, logReturns_HL_BSESN_Weekly, "BSE", "Weekly")

cal_latexRow_BSESN_Weekly = vec(hcat(round.(cal_lowerCI_BSESN_Weekly, digits = 4), calMom_BSESN_Weekly, round.(cal_upperCI_BSESN_Weekly, digits = 4))')
cal_latex_df_BSESN_Weekly = DataFrame(cal_latexRow_BSESN_Weekly', :auto)
cal_latex_output_BSESN_Weekly = pretty_table(cal_latex_df_BSESN_Weekly, backend = :latex)

#####################################################################

