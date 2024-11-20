
##### Gaunersdorfer & Hommes 2000 Agent Based Model 

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

function ghABM(Time, n)

    ### Parameters

    # Set Seed for Reproducibility
    Random.seed!(1234)

    # Number of Timesteps
    T = Time

    # Dividend Mean
    yBar = 1

    # Risk Free Rate
    r = 0.001
    
    # Strength of Fundamentalists Mean Reversion Beliefs
    nu = 1

    # Strength of Trend-Following Chartists Technical Beliefs
    g = 2

    sigmaDelta = 1
    sigmaSq = 1
    sigmaEps = 1
    
    # Strength of Memory
    eta = 0

    # Speculators Sensitivity to Mispricing
    alpha = 2000
    
    # Intensity of Choice
    beta = 2

    # Risk Aversion
    lambda = 1/sigmaSq

    # Fundamental Value
    pStar = yBar/r

    ### Initialise Variables and Matrices

    # Dividends of the Risky Asset
    dividends = zeros(T)
    
    # Prices of the Risky Asset
    price = zeros(T)

    # Total Returns of Risky Assets
    returns = zeros(T)

    # Fundamentalists Expected Return of the Risky Asset
    expRet_Fund = zeros(T)

    # Chartists Expected Return of the Risky Asset
    expRet_Chart = zeros(T)

    # Fundamentalists Demand of the Risky Asset
    demand_Fund = zeros(T)

    # Chartists Demand of the Risky Asset
    demand_Chart = zeros(T)

    # Accumulated Profits by Fundamentalists
    accProf_Fund = zeros(T)

    # Accumulated Profits by Chartists
    accProf_Chart = zeros(T)

    # Percentage of Fundamentalists
    n_Fund = zeros(T)

    # Percentage of Chartists
    n_Chart = zeros(T)

    for i in 1:2
        dividends[i] = yBar
        price[i] = 1000
        expRet_Fund[i] = 1000
        expRet_Chart[i] = 1000
    end

    for t in 3:T

        # Delta Error Term
        delta = rand(Normal(0, sigmaDelta), 1)[1]

        dividends[t] = yBar .+ delta

        # Fundamentalists Expected Return at time t+1
        expRet_Fund[t] = pStar + (nu * (price[t-1] - pStar))

        # Chartists Expected Return at time t+1
        expRet_Chart[t] = price[t-1] + (g * (price[t-1] - price[t-2]))

        # Chartists Share of the Risky Asset market at time t
        n_Chart[t] = (1/(1 + exp(beta * (accProf_Fund[t-1] - accProf_Chart[t-1])))) * exp(-((pStar - price[t-1])^2)/alpha)

        # Fundamentalists Share of the Risky Asset market at time t
        n_Fund[t] = 1 - n_Chart[t]

        # Sigma Error Term
        epsilon = rand(Normal(0, sigmaEps), 1)[1]

        # Price of the Risky Asset at time t
        price[t] = (1/(1 + r)) * ((n_Chart[t] * expRet_Chart[t]) + 
                   (n_Fund[t] * expRet_Fund[t]) + 
                    yBar) .+ epsilon

        # Fundamentalists Demand of the Risky Asset at time t
        demand_Fund[t] = (expRet_Fund[t] + dividends[t] - (1 + r) * price[t]) / (lambda * sigmaSq)

        # Chartists Demand of the Risky Asset at time t
        demand_Chart[t] = (expRet_Chart[t] + dividends[t] - (1 + r) * price[t]) / (lambda * sigmaSq)

        # Accumulated Profits by Fundamentalists at time t
        accProf_Fund[t] = (price[t] + dividends[t] - (1 + r) * price[t-1]) * demand_Fund[t-1] + 
        (eta * accProf_Fund[t-1])

        # Accumulated Profits by Chartists at time t
        accProf_Chart[t] = (price[t] + dividends[t] - (1 + r) * price[t-1]) * demand_Chart[t-1] + 
        (eta * accProf_Chart[t-1])

        returns[t] = ((price[t] - price[t-1]) / price[t-1]) + (dividends[t] / price[t-1])
    end

    n1 = n - 1
    p = [mean(price[i:i+n1]) for i in 1:n:length(price) if i+n1 <= length(price)]
    div = [mean(dividends[i:i+n1]) for i in 1:n:length(dividends) if i+n1 <= length(dividends)]

    r = zeros(T)

    for t in 2:665

        r[t] = ((p[t] - p[t-1]) / p[t-1]) + (div[t] / p[t-1])

    end
    
    return p, r

end

timeEnd = 13300

prices, returns = ghABM(timeEnd, 20)

###############################################################################

plotStart = 101
plotEnd = 665

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

function plotPrices(Prices, bt, et)

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

    plot(p1, jse, layout = (2, 1), size = (900, 900), 
         margin = 2mm)

end

display(plotPrices(prices, plotStart, plotEnd))

###############################################################################

# Asset Return Plots

function plotReturns(Returns, bt, et)

    t = bt:et

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

    hline!([mean(Returns)], label = round(mean(Returns), digits = 4), 
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

    a1Density = pdf.(Normal(mean(Returns), std(Returns)), xVals)
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

# For Loop Plots

gs = 2:0.5:6

nu = 1
g = 2

for g in gs

    prices, returns = ghABM(timeEnd, nu, g)

    function plotPrices(Prices, bt, et)

            t = bt:et
        
            jse = plot(1:lengthJSE, weeklyData, label = "JSE Top 40", title = "JSE Top 40 Index", 
                    xlabel = "Week", ylabel = "Price", legend = false, 
                    yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
                    tick_direction = :none, color = "purple1", lw = 1.5, 
                    gridlinewidth = 1.5, gridstyle = :dash)
        
            p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset, g = $g", 
                    xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
                    tick_direction = :none, color = "darkorange2", lw = 1.5, 
                    gridlinewidth = 1.5, gridstyle = :dash)
        
            plot(p1, jse, layout = (2, 1), size = (900, 900), 
                margin = 2mm)
        
    end

    function plotReturns(Returns, bt, et)

            t = bt:et
        
            jse = plot(1:lengthJSE, returnsJSE, title = "JSE Top 40 Index", label = false, 
                    xlabel = "Week", ylabel = "Return", legend = :topleft, framestyle = :box, 
                    tick_direction = :none, color = "purple1", ylim = (-0.15, 0.25), 
                    grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)
        
            hline!([mean(returnsJSE)], label = round(mean(returnsJSE), digits = 4), 
                    color =:black, lw = 1, linestyle =:dash)
        
            p1 = plot(t, Returns[t], label = false, title = "Risky Asset, g = $g", 
                    xlabel = "Week", ylabel = "Returns", legend = :topleft, framestyle = :box, 
                    tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
                    grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)
        
            hline!([mean(Returns)], label = round(mean(Returns), digits = 4), 
                    color =:black, lw = 1, linestyle =:dash)
        
            plot(p1, jse, layout = (2, 1), size = (900, 900), 
            margin = 2mm)
        
    end

    display(plotPrices(prices, plotStart, plotEnd))
    display(plotReturns(returns, plotStart, plotEnd))

end