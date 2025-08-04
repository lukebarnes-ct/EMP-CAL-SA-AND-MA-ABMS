
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

    return price, fundValue, returns, demand_Fund, demand_Chart, expRet_Fund, expRet_Chart, excGain

end

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
        # returns[t] = log(price[t]/price[t-1])
    end
    
    return price, fundValue, returns, n_Fund, n_Chart, 
    demand_Fund, demand_Chart, relFit

end

# Return Required Moments

function teststatistic(x)
    n = x.n_x*x.n_y/(x.n_x+x.n_y)
    sqrt(n)*x.δ
end

function f_HL(x, repetitions, index, timescale)

    simMom = getSimulatedMoments(x, repetitions, "H&L", index, timescale)
    obj = getObjectiveFunction(momentsJSE_Daily, simMom, bootstrapMatrixJSE_Daily)

    return obj[1]
end

function f_FW(x, repetitions, index, timescale)

    simMom = getSimulatedMoments(x, repetitions, "F&W", index, timescale)
    obj = getObjectiveFunction(momentsJSE_Daily, simMom, bootstrapMatrixJSE_Daily)

    return obj[1]
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
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsSSE_Daily)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            elseif timescale == "Weekly"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsSSE_Weekly)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            end
    
        elseif index == "BSE"
    
            if timescale == "Daily"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsBSE_Daily)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            elseif timescale == "Weekly"
    
                ks_test = ApproximateTwoSampleKSTest(Returns[t], returnsBSE_Weekly)
                U[5] = round(teststatistic(ks_test), digits = 4)
    
            end
    
        end

    else

        U[5] = 0
    end

    U[6] = round(hurst_exponent(Returns[t], 1:19)[1], digits = 4)

    return U

end

function getSimulatedMoments(par, N, ABM, index, timescale)
    
    timeBegin = 1
    timeEnd = 10000

    sMoments = zeros(6, N)

    if ABM == "H&L"

        MU = par[1]
        GAMMA = par[2]
        DELTA = par[3] 
        ALPHA = par[4]

        for n in 1:N

            prices, fv, returns, demFund, demChart, 
            expFund, expChart, exG = hlABM(timeEnd, n, 20, 20, MU, GAMMA, DELTA, ALPHA)

            moments = getMoments(returns, timeBegin, timeEnd, "Simulated", index, timescale)

            sMoments[:, n] = moments
        end

        simMoments = mean(sMoments, dims = 2)

    elseif ABM == "F&W"

        BETA = par[1]
        CHI = par[2]
        PHI = par[3]
        SIGMA_C = par[4]
        SIGMA_F = par[5]
        ALPHA_0 = par[6]
        ALPHA_N = par[7]
        ALPHA_P = par[8]

        for n in 1:N

            prices, fv, returns, demFund, demChart, 
            expFund, expChart, exG = fwABM(timeEnd, n, BETA, CHI, PHI, SIGMA_C, SIGMA_F, ALPHA_0, ALPHA_N, ALPHA_P)

            moments = getMoments(returns, timeBegin, timeEnd, "Simulated", index, timescale)

            sMoments[:, n] = moments
        end

        simMoments = mean(sMoments, dims = 2)

    elseif ABM == "XU"

        

    end

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

f_HL_MBB(x, grad) = f_HL(x, repetitions, index, timescale)
f_FW_MBB(x, grad) = f_FW(x, repetitions, index, timescale)

function nelderMeadSimulation(ABM, threshold)
    
    if ABM == "H&L"

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

        perturb = 0.05

        counter = 1

        while threshold > minThreshold

            println("Counter: $counter")

            newParameters = [x * (1 + rand(MersenneTwister(counter), Uniform(-perturb, perturb))) for x in bestParameters]
            newParameters = [clamp(x, lower, upper) for (x, lower, upper) in zip(newParameters, lowerBounds, upperBounds)]

            optCurrent = Opt(:LN_NELDERMEAD, length(newParameters))
            optCurrent.xtol_rel = 1e-6
            optCurrent.lower_bounds = lowerBounds
            optCurrent.upper_bounds = upperBounds
            optCurrent.min_objective = f_HL_MBB

            (currentValue, currentParameters, ret) = NLopt.optimize(optCurrent, newParameters)

            if (currentValue < bestValue) || (currentValue - bestValue < threshold)

                println("NEW BEST PARAMETERS: $currentParameters")
                println("NEW BEST VALUE: $currentValue")

                bestValue = currentValue
                bestParameters = currentParameters
            end

            threshold = threshold * 0.95
            counter = counter + 1
        end

    elseif ABM == "F&W"

        lowerBounds = [0, 0, 0, 0, 0, 0, -100, 0, 0.01]
        upperBounds = [0.1, 1, 10, 1000, 1000, 1000, 100, 100, 100]

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

        perturb = 0.05

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
        

    elseif ABM == "XU"

        

    end

    return bestParameters, bestValue
end

#####################################################################

momentsJSE_Daily = getMoments(returnsJSE_Daily, 1, lengthJSE_Daily, "Real", "JSE", "Daily")
momentsJSE_Weekly = getMoments(returnsJSE_Weekly, 1, lengthJSE_Weekly, "Real", "JSE", "Weekly")

momentsSSE50_Daily = getMoments(returnsJSE_Daily, 1, lengthSSE50_Daily, "Real", "SSE50", "Daily")
momentsSSE50_Weekly = getMoments(returnsJSE_Weekly, 1, lengthSSE50_Weekly, "Real", "SSE50", "Weekly")

momentsBSESN_Daily = getMoments(returnsJSE_Daily, 1, lengthBSESN_Daily, "Real", "BSESN", "Daily")
momentsBSESN_Weekly = getMoments(returnsJSE_Weekly, 1, lengthBSESN_Weekly, "Real", "BSESN", "Weekly")

blockWindow = 100
blockSamples = 1000

bootstrapMatrixJSE_Daily = getMovingBlockBootstrapMatrix(2001, returnsJSE_Daily, blockWindow, blockSamples)
bootstrapMatrixJSE_Weekly = getMovingBlockBootstrapMatrix(2002, returnsJSE_Weekly, blockWindow, blockSamples)

bootstrapMatrixSSE50_Daily = getMovingBlockBootstrapMatrix(2003, returnsSSE50_Daily, blockWindow, blockSamples)
bootstrapMatrixSSE50_Weekly = getMovingBlockBootstrapMatrix(2004, returnsSSE50_Weekly, blockWindow, blockSamples)

bootstrapMatrixBSESN_Daily = getMovingBlockBootstrapMatrix(2005, returnsBSESN_Daily, blockWindow, blockSamples)
bootstrapMatrixBSESN_Weekly = getMovingBlockBootstrapMatrix(2006, returnsBSESN_Weekly, blockWindow, blockSamples)

simMomentsJSE_Daily = getSimulatedMoments([2.1652601173749995, 0.14593493917215378, 1.0, 0.9372527766928516], 100, "H&L", "JSE", "Daily")
objJSE_Daily = getObjectiveFunction(momentsJSE_Daily, simMomentsJSE_Daily, bootstrapMatrixJSE_Daily)

repetitions = 10
index = "JSE"
timescale = "Daily"

optParam_HL_JSE_Daily, optValue_HL_JSE_Daily = nelderMeadSimulation("H&L", 1)

optParam_FW_JSE_Daily, optValue_FW_JSE_Daily = nelderMeadSimulation("F&W", 0.00001)

### Simulation works for H&L - implement for other ABMs
### Implemented for F&W - simulating gets stuck on Counter 19

par = [0.09624890809011971, 0.45302257644532445, 9.707580471090601, 0.5508945869462822, 6.4414006371501245, 758.1136841678771, 29.113874289932546, 88.86763965754545]

bestOBJ_JSE = f_FW(par, repetitions, index, timescale)

simMom_FW = getSimulatedMoments(par, repetitions, "F&W", index, timescale)

prices, fv, returns, demFund, demChart, 
            expFund, expChart, exG = fwABM(10000, repetitions, 0.09624890809011971, 0.45302257644532445, 9.707580471090601, 0.5508945869462822, 6.4414006371501245, 758.1136841678771, 29.113874289932546, 88.86763965754545)

any(isnan, prices)

### There is seemingly an issue with the output of this model using these Parameters
### Probably need to further constrain parameters, experiment with this.

### Seemingly stumbled upon results that work F&W and calibrating to the Daily JSE Returns