
##### He and Li 2008 Agent Based Model 

using Random
using Plots
using Distributions
using Optim
using NLsolve
using JLD2
using Base.Threads
using Printf
using Plots.PlotMeasures
using StatsBase
using TypedTables
using StatsPlots

# Load JSE Top 40 Index from .jld2 file
@load "Data/jsetop40.jld2" weeklyData

#################################################################################

# Calculate the returns of the JSE Top 40 Price Data

lengthJSE = length(weeklyData)
returnsJSE = zeros(lengthJSE)

for i in 2:lengthJSE

    returnsJSE[i] = ((weeklyData[i] - weeklyData[i-1]) ./ weeklyData[i-1])

end

#################################################################################

function hlABM(Time, n1, n2, mu, gamma, delta, alpha)

    ### Parameters

    # Set Seed for Reproducibility
    Random.seed!(1234)

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
    b = b2/sigma_C

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

        # Chartists Expected Return at time t + 1
        expRet_Chart[t] = (gamma * (price[t] - u[t])) - ((R - 1) * (price[t] - in_Fund))

        # Chartists Expected Variance at time t + 1
        expVar_Chart[t] = sigma_C * (1 + q + (b * v[t]))

        # Chartists Demand at time t
        demand_Chart[t] = (expRet_Chart[t]) / (a2 * expVar_Chart[t])

        # Sample Mean at time t
        u[t] = (delta * u[t-1]) + ((1 - delta) * price[t])

        # Sample Variance at time t
        v[t] = (delta * v[t-1]) + (delta * (1 - delta) * (price[t] - u[t-1])^2)
    end

    return price, fundValue, returns, demand_Fund, demand_Chart

end

###############################################################################

# Review Output from model and FINAL Parameters

timeEnd = 10000

plotStart = 1001
plotEnd = 1565

### Hyperparameters

# Market Maker Price Adjustment Speed
MU = 2

# Chartist Extrapolation Rate
GAMMA = 0.3

# Delta
DELTA = 0.85

# Fundamentalist Price Adjustment Speed
ALPHA = 0.1

prices, fv, returns, demFund, demChart = hlABM(timeEnd, 20, 20, MU, GAMMA, DELTA, ALPHA)

###############################################################################

# Descriptive Statistics of Asset Returns

function descriptiveStatistics(Returns, bt, et)

    t = bt:et

    Mean = round(mean(Returns), digits = 4)
    jseMean = round(mean(returnsJSE), digits = 4)

    Median = round(median(Returns), digits = 4)
    jseMedian = round(median(returnsJSE), digits = 4)

    SD = round(std(Returns), digits = 4)
    jseSD = round(std(returnsJSE), digits = 4)

    Skewness = round(skewness(Returns), digits = 4)
    jseSkewness = round(skewness(returnsJSE), digits = 4)

    Kurtosis = round(kurtosis(Returns), digits = 4)
    jseKurtosis = round(kurtosis(returnsJSE), digits = 4)

    descStat = Table(Asset = ["1", "JSE"], 
                     Mean = [Mean, jseMean],
                     Median = [Median, jseMedian],
                     SD = [SD, jseSD],
                     Skewness = [Skewness, jseSkewness],
                     Kurtosis = [Kurtosis, jseKurtosis])
    
    return descStat
end

returnStatistics = descriptiveStatistics(returns, plotStart, plotEnd)

###############################################################################

# Price Plots

function plotPrices(Prices, FValue, bt, et)

    t = bt:et

    jse = plot(1:lengthJSE, weeklyData, label = "JSE Top 40", title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "purple1", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

    p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset", 
              xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", lw = 1.5, 
              gridlinewidth = 1.5, gridstyle = :dash)

    pf =  plot(t, FValue[t], label = "Fundamental Value",
    xlabel = "Week", ylabel = "FV", legend = false, framestyle = :box, 
    tick_direction = :none, color = "red", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    plot(p1, pf, jse, layout = (3, 1), size = (900, 900), 
         margin = 2mm)

end

display(plotPrices(prices, fv, plotStart, plotEnd))

###############################################################################

# Asset Return Plots

function plotReturns(Returns, bt, et)

    t = bt:et
    m = mean(Returns[t])

    jse = plot(1:lengthJSE, returnsJSE, title = "JSE Top 40 Index", label = false, 
               xlabel = "Week", ylabel = "Return", legend = :topleft, framestyle = :box, 
               tick_direction = :none, color = "purple1", ylim = (-0.15, 0.25), 
               grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

    hline!([mean(returnsJSE)], label = round(mean(returnsJSE), digits = 4), 
            color =:black, lw = 1, linestyle =:dash)

    p1 = plot(t, Returns[t], label = false, title = "Risky Asset", 
              xlabel = "Week", ylabel = "Returns", legend = :topleft, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
              grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

    hline!([m], label = round(m, digits = 4), 
            color =:black, lw = 1, linestyle =:dash)

    plot(p1, jse, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturns(returns, plotStart, plotEnd))

###############################################################################

# Return Distribution Histogram and Density Plots

function plotReturnDistribution(Returns, bt, et)

    t = bt:et

    xVals = range(-0.15, stop = 0.15, length = 60)
    binSize = xVals

    jse = histogram(returnsJSE, bins = binSize, title = "JSE Top 40 Index",
    xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
    label = false, color = "purple1", alpha = 0.5, framestyle = :box, 
    tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
    gridalpha = 0.125)

    jseDensity = pdf.(Normal(mean(returnsJSE), std(returnsJSE)), xVals)

    plot!(xVals, jseDensity, label = false, lw = 2.5, color = "purple1")

    p1 = histogram(Returns[t], bins = binSize, title = "Risky Asset", 
    xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
    label = false, color = "darkorange2", alpha = 0.5, framestyle = :box, 
    tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
    gridalpha = 0.125)

    a1Density = pdf.(Normal(mean(Returns[t]), std(Returns[t])), xVals)
    plot!(xVals, a1Density, label = false, lw = 2.5, color = "darkorange2")

    plot(p1, jse, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturnDistribution(returns, plotStart, plotEnd))

###############################################################################

# Return AutoCorrelation Plots

function plotAutoCorrelations(Returns, bt, et)

    t = bt:et

    wt = length(weeklyData)

    maxLags = 25
    xVals = 0:maxLags

    sqCol = "orchid"

    jseRetAuto = autocor(returnsJSE)[1:(maxLags+1)]
    jseAbsRetAuto = autocor(abs.(returnsJSE))[1:(maxLags+1)]

    jse = plot(xVals, jseRetAuto, seriestype = :line, title = "JSE Top 40 Index",
    xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
    tick_direction = :none, color = "purple1", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

    hline!([0], label = false, color =:black, lw = 0.75)

    scatter!(xVals, jseRetAuto, marker = (:circle, 4, stroke(:purple1)), 
    color = "purple1", label = "Returns")

    plot!(xVals, jseAbsRetAuto, seriestype = :line, label = false, 
    color = sqCol, lw = 1.5)

    scatter!(xVals, jseAbsRetAuto, marker = (:square, 4, stroke(sqCol)), 
    color = sqCol, label = "|Returns|")

    a1RetAuto = autocor(Returns[t])[1:(maxLags+1)]
    a1AbsRetAuto = (autocor(abs.(Returns[t]))[1:(maxLags+1)])

    p1 = plot(xVals, a1RetAuto, seriestype = :line, title = "Risky Asset",
    xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
    tick_direction = :none, color = "darkorange2", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

    scatter!(xVals, a1RetAuto, marker = (:circle, 4, stroke(:darkorange2)), 
    color = "darkorange2", label = "Returns")

    hline!([0], label = false, color =:black, lw = 0.75)

    plot!(xVals, a1AbsRetAuto, seriestype = :line, label = false, 
    color = sqCol, lw = 1.5)

    scatter!(xVals, a1AbsRetAuto, marker = (:square, 4, stroke(sqCol)), 
    color = sqCol, label = "|Returns|")

    plot(p1, jse, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotAutoCorrelations(returns, plotStart, plotEnd))

###############################################################################