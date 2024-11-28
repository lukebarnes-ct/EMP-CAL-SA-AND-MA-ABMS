
##### Franke & Westerhoff 2012 Agent Based Model 

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

function fwABM(Time)

    ### Parameters

    # Set Seed for Reproducibility
    Random.seed!(1234)

    # Number of Timesteps
    T = Time

    mu = 0.01
    beta = 1
    chi = 1.5
    phi = 0.12
    sigma_C = 2.087
    sigma_F = 0.758
    alpha_0 = -0.327
    alpha_N = 1.79
    alpha_P = 18.43

    # Dividend Growth Rate
    phi = 0.002
    phiSD = 0.015

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

###############################################################################

# Review Output from model and FINAL Parameters

timeEnd = 10000

plotStart = 8501
plotEnd = 9065

prices, fv, returns, numFund, numChart, 
demFund, demChart, relativeFitness = fwABM(timeEnd)

display(plotPrices(prices, fv, plotStart, plotEnd))
display(plotReturns(returns, plotStart, plotEnd))

function printOutput(bt, et, type)

    head = ["$bt", "$bt+1", "$bt+2", "$bt+3", "$bt+4", "$et-4", "$et-3", "$et-2", "$et-1", "$et"]

    lt = length(bt:et)

    if type == "Prop"
        println("Fundamentalists Proportion")
        pretty_table(transpose(numFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Proportion")
        pretty_table(transpose(numChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)

    elseif type == "Demand"
        println("Fundamentalists Demand")
        pretty_table(transpose(demFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Demand")
        pretty_table(transpose(demChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)

    elseif type == "Price"
        println("Price")
        pretty_table(transpose(prices[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Return")
        pretty_table(transpose(returns[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
    end
end

printOutput(plotStart, plotEnd, "Prop")
printOutput(plotStart, plotEnd, "Demand")
printOutput(plotStart, plotEnd, "Price")

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

    p1 = plot(t, exp.(Prices[t]), label = "Price", title = "Risky Asset", 
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

plot(plotStart:plotEnd, numChart[plotStart:plotEnd], label = "Price", title = "Number of Fundamentalists", 
              xlabel = "Week", ylabel = "Fundamentalists (%)", legend = false, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", lw = 1.5, 
              gridlinewidth = 1.5, gridstyle = :dash)