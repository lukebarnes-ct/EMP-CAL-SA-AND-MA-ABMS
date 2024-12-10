
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
using LaTeXStrings
using Subscripts

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

#################################################################################

function ghABM(Time, p1)

    ### Parameters

    # Set Seed for Reproducibility
    Random.seed!(1234)

    # Number of Timesteps
    T = Time

    # Risk Free Rate
    r = 0.002
    
    # Strength of Fundamentalists Mean Reversion Beliefs
    nu = 1

    # Strength of Trend-Following Chartists Technical Beliefs
    g = 2

    sigmaDelta = 1
    sigmaSq = 1
    sigmaEps = 1
    
    # Strength of Memory
    eta = 0.5

    # Speculators Sensitivity to Mispricing
    alpha = 2000
    
    # Intensity of Choice
    beta = 2

    # Risk Aversion
    lambda = 1/sigmaSq

    ### Initialise Variables and Matrices
    
    # Prices of the Risky Asset
    price = zeros(T)

    # Total Returns of Risky Assets
    returns = zeros(T)

    fundValue = zeros(T)

    # Dividends of the Risky Asset
    dividends = zeros(T)

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
        dividends[i] = p1 * 0.001
        price[i] = p1
        fundValue[i] = p1
        expRet_Fund[i] = p1
        expRet_Chart[i] = p1
    end

    for t in 3:T

        fundValue[t] =  fundValue[t-1]

        # Delta Error Term
        delta = rand(Normal(0, sigmaDelta), 1)[1]

        dividends[t] =  1 + delta

        # Fundamentalists Expected Return at time t+1
        expRet_Fund[t] = fundValue[t] + (nu * (price[t-1] - fundValue[t]))

        # Chartists Expected Return at time t+1
        expRet_Chart[t] = price[t-1] + (g * (price[t-1] - price[t-2]))

        # Chartists Share of the Risky Asset market at time t
        n_Chart[t] = (1/(1 + exp(beta * (accProf_Fund[t-1] - accProf_Chart[t-1])))) * exp(-((fundValue[t] - price[t-1])^2)/alpha)

        # Fundamentalists Share of the Risky Asset market at time t
        n_Fund[t] = 1 - n_Chart[t]

        # Sigma Error Term
        epsilon = rand(Normal(0, sigmaEps), 1)[1]

        # Price of the Risky Asset at time t
        price[t] = (1/(1 + r)) * ((n_Chart[t] * expRet_Chart[t]) + 
                   (n_Fund[t] * expRet_Fund[t]) + 
                    1) .+ epsilon

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
    
    return price, fundValue, returns, expRet_Fund, expRet_Chart, 
    n_Fund, n_Chart, demand_Fund, demand_Chart, accProf_Fund, accProf_Chart

end

###############################################################################

# Review Output from model and FINAL Parameters

timeEnd = 13300

plotStart_Daily = 1001
plotEnd_Daily_JSE = plotStart_Daily - 1 + lengthJSE_Daily 
plotEnd_Daily_SSE50 = plotStart_Daily - 1 + lengthSSE50_Daily
plotEnd_Daily_BSESN = plotStart_Daily - 1 + lengthBSESN_Daily

plotStart_Weekly = 101
plotEnd_Weekly_JSE = plotStart_Weekly - 1 + lengthJSE_Weekly 
plotEnd_Weekly_SSE50 = plotStart_Weekly - 1 + lengthSSE50_Weekly
plotEnd_Weekly_BSESN = plotStart_Weekly - 1 + lengthBSESN_Weekly


prices, fv, returns, expFund, expChart, numFund, numChart, 
demFund, demChart, aProfFund, aProfChart = ghABM(timeEnd, 11500)

function printOutput(bt, et, type)

    head = ["$bt", "$bt+1", "$bt+2", "$bt+3", "$bt+4", "$et-4", "$et-3", "$et-2", "$et-1", "$et"]

    lt = length(bt:et)

    if type == "ER"
        println("Fundamentalists Expected Return")
        pretty_table(transpose(expFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Expected Return")
        pretty_table(transpose(expChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)

    elseif type == "Prop"
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
    elseif type == "Profit"
        println("Fundamentalists Profit")
        pretty_table(transpose(aProfFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
        header = head)
        println("Chartists Profit")
        pretty_table(transpose(aProfChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
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

printOutput(plotStart, plotEnd, "ER")
printOutput(plotStart, plotEnd, "Prop")
printOutput(plotStart, plotEnd, "Demand")
printOutput(plotStart, plotEnd, "Profit")
printOutput(plotStart, plotEnd, "Price")

###############################################################################

# Descriptive Statistics of Asset Returns

function descriptiveStatistics(Returns, bt, et, indexReturns)

    t = bt:et

    Mean = round(mean(Returns[t]), digits = 4)
    indexMean = round(mean(indexReturns), digits = 4)

    Median = round(median(Returns[t]), digits = 4)
    indexMedian = round(median(indexReturns), digits = 4)

    SD = round(std(Returns[t]), digits = 4)
    indexSD = round(std(indexReturns), digits = 4)

    Skewness = round(skewness(Returns[t]), digits = 4)
    indexSkewness = round(skewness(indexReturns), digits = 4)

    Kurtosis = round(kurtosis(Returns[t]), digits = 4)
    indexKurtosis = round(kurtosis(indexReturns), digits = 4)

    descStat = Table(Asset = ["1", "Index"], 
    Mean = [Mean, indexMean],
    Median = [Median, indexMedian],
    SD = [SD, indexSD],
    Skewness = [Skewness, indexSkewness],
    Kurtosis = [Kurtosis, indexKurtosis])

    return descStat
end

returnStatisticsJSE = descriptiveStatistics(returns, plotStart_Daily, plotEnd_Daily_JSE, returnsJSE_Daily)
returnStatisticsSSE = descriptiveStatistics(returns, plotStart_Daily, plotEnd_Daily_SSE50, returnsSSE50_Daily)
returnStatisticsBSE = descriptiveStatistics(returns, plotStart_Daily, plotEnd_Daily_BSESN, returnsBSESN_Daily)

returnStatisticsJSE_Weekly = descriptiveStatistics(returns, plotStart_Weekly, plotEnd_Weekly_JSE, returnsJSE_Weekly)
returnStatisticsSSE_Weekly = descriptiveStatistics(returns, plotStart_Weekly, plotEnd_Weekly_SSE50, returnsSSE50_Weekly)
returnStatisticsBSE_Weekly = descriptiveStatistics(returns, plotStart_Weekly, plotEnd_Weekly_BSESN, returnsBSESN_Weekly)

###############################################################################

# Price Plots

default(fontfamily = "ComputerModern")

function plotPrices(Prices, FValue, bt, et, index, timescale)

    t = bt:et

    if index == "JSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthJSE_Daily, daily_JSETOP40_Data, label = "JSE Top 40", title = "JSE Top 40 Index", 
            xlabel = "Day", ylabel = "Price", legend = false, 
            yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
            tick_direction = :none, color = "purple1", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthJSE_Weekly, weekly_JSETOP40_Data, label = "JSE Top 40", title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "purple1", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

        end

    elseif index == "SSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthSSE50_Daily, daily_SSE50_Data, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Day", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "dodgerblue4", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthSSE50_Weekly, weekly_SSE50_Data, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "dodgerblue4", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

        end

    elseif index == "BSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthBSESN_Daily, daily_BSESN_Data, label = "BSE Sensex", title = "BSE Sensex Index", 
               xlabel = "Day", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "deeppink2", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthBSESN_Weekly, weekly_BSESN_Data, label = "BSE Sensex", title = "BSE Sensex Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "deeppink2", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

        end

    end

    p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset", 
              xlabel = "Day", ylabel = "Price", legend = false, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", lw = 1.5, 
              gridlinewidth = 1.5, gridstyle = :dash)

    pf =  plot(t, FValue[t], label = "Fundamental Value",
    xlabel = "Week", ylabel = "FV", legend = false, framestyle = :box, 
    tick_direction = :none, color = "red", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash)

    plot(p1, pf, indexPlot, layout = (3, 1), size = (900, 900), 
         margin = 2mm)

end

display(plotPrices(prices, fv, plotStart, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotPrices(prices, fv, plotStart, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotPrices(prices, fv, plotStart, plotEnd_Daily_BSESN, "BSE", "Daily"))

###############################################################################

# Asset Return Plots

function plotReturns(Returns, bt, et, index, timescale)

    t = bt:et

    if index == "JSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthJSE_Daily, returnsJSE_Daily, label = "JSE Top 40", title = "JSE Top 40 Index", 
            xlabel = "Day", ylabel = "Return", legend = :topleft, 
            framestyle = :box, tick_direction = :none, color = "purple1", 
            ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

            hline!([mean(returnsJSE_Daily)], label = round(mean(returnsJSE_Daily), digits = 4), 
            color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthJSE_Weekly, returnsJSE_Weekly, label = "JSE Top 40", title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "purple1", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsJSE_Weekly)], label = round(mean(returnsJSE_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        end

    elseif index == "SSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthSSE50_Daily, returnsSSE50_Daily, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Day", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "dodgerblue4", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsSSE50_Daily)], label = round(mean(returnsSSE50_Daily), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthSSE50_Weekly, returnsSSE50_Weekly, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "dodgerblue4", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsSSE50_Weekly)], label = round(mean(returnsSSE50_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        end

    elseif index == "BSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthBSESN_Daily, returnsBSESN_Daily, label = "BSE Sensex", title = "BSE Sensex Index", 
               xlabel = "Day", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "deeppink2", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsBSESN_Daily)], label = round(mean(returnsBSESN_Daily), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthBSESN_Weekly, returnsBSESN_Weekly, label = "BSE Sensex", title = "BSE Sensex Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "deeppink2", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsBSESN_Weekly)], label = round(mean(returnsBSESN_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        end

    end

    p1 = plot(t, Returns[t], label = false, title = "Risky Asset", 
              xlabel = "Week", ylabel = "Returns", legend = :topleft, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
              grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

    hline!([mean(Returns[t])], label = round(mean(Returns[t]), digits = 4), 
            color =:black, lw = 1, linestyle =:dash)

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturns(returns, plotStart, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotReturns(returns, plotStart, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotReturns(returns, plotStart, plotEnd_Daily_BSESN, "BSE", "Daily"))

###############################################################################

# Return Distribution Histogram and Density Plots

function plotReturnDistribution(Returns, bt, et, index, timescale)

    t = bt:et
    xVals = range(-0.15, stop = 0.15, length = 60)
    binSize = xVals

    if index == "JSE"

        if timescale == "Daily"

            indexPlot = histogram(returnsJSE_Daily, bins = binSize, title = "JSE Top 40 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "purple1", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsJSE_Daily), std(returnsJSE_Daily)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "purple1")

        elseif timescale == "Weekly"

            indexPlot = histogram(returnsJSE_Weekly, bins = binSize, title = "JSE Top 40 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "purple1", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsJSE_Weekly), std(returnsJSE_Weekly)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "purple1")

        end

    elseif index == "SSE"

        if timescale == "Daily"

            indexPlot = histogram(returnsSSE50_Daily, bins = binSize, title = "SSE 50 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "dodgerblue4", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsSSE50_Daily), std(returnsSSE50_Daily)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "dodgerblue4")

        elseif timescale == "Weekly"

            indexPlot = histogram(returnsSSE50_Weekly, bins = binSize, title = "SSE 50 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "dodgerblue4", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsSSE50_Weekly), std(returnsSSE50_Weekly)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "dodgerblue4")

        end

    elseif index == "BSE"

        if timescale == "Daily"

            indexPlot = histogram(returnsBSESN_Daily, bins = binSize, title = "BSE Sensex Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "deeppink2", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsBSESN_Daily), std(returnsBSESN_Daily)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "deeppink2")

        elseif timescale == "Weekly"

            indexPlot = histogram(returnsBSESN_Weekly, bins = binSize, title = "BSE Sensex Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "deeppink2", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

            densityPlot = pdf.(Normal(mean(returnsBSESN_Weekly), std(returnsBSESN_Weekly)), xVals)

            plot!(xVals, densityPlot, label = false, lw = 2.5, color = "deeppink2")

        end

    end

    p1 = histogram(Returns[t], bins = binSize, title = "Risky Asset", 
    xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
    label = false, color = "darkorange2", alpha = 0.5, framestyle = :box, 
    tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
    gridalpha = 0.125)

    a1Density = pdf.(Normal(mean(Returns[t]), std(Returns[t])), xVals)
    plot!(xVals, a1Density, label = false, lw = 2.5, color = "darkorange2")

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturnDistribution(returns, plotStart, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotReturnDistribution(returns, plotStart, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotReturnDistribution(returns, plotStart, plotEnd_Daily_BSESN, "BSE", "Daily"))

###############################################################################

# Return AutoCorrelation Plots


function plotAutoCorrelations(Returns, bt, et, index, timescale)

    t = bt:et
    maxLags = 25
    xVals = 0:maxLags

    sqCol = "green4"
    absCol = "firebrick"

    if index == "JSE"

        if timescale == "Daily"

            retAuto = autocor(returnsJSE_Daily)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsJSE_Daily))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsJSE_Daily.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "JSE Top 40 Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "purple1", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:purple1)), 
            color = "purple1", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        elseif timescale == "Weekly"

            retAuto = autocor(returnsJSE_Weekly)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsJSE_Weekly))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsJSE_Weekly.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "JSE Top 40 Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "purple1", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:purple1)), 
            color = "purple1", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        end

    elseif index == "SSE"

        if timescale == "Daily"

            retAuto = autocor(returnsSSE50_Daily)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsSSE50_Daily))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsSSE50_Daily.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "SSE 50 Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "dodgerblue4", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:dodgerblue4)), 
            color = "dodgerblue4", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        elseif timescale == "Weekly"

            retAuto = autocor(returnsSSE50_Weekly)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsSSE50_Weekly))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsSSE50_Weekly.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "SSE 50 Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "dodgerblue4", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:dodgerblue4)), 
            color = "dodgerblue4", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        end

    elseif index == "BSE"

        if timescale == "Daily"

            retAuto = autocor(returnsBSESN_Daily)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsBSESN_Daily))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsBSESN_Daily.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "BSE Sensex Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "deeppink2", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:deeppink2)), 
            color = "deeppink2", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        elseif timescale == "Weekly"

            retAuto = autocor(returnsBSESN_Weekly)[1:(maxLags+1)]
            absRetAuto = autocor(abs.(returnsBSESN_Weekly))[1:(maxLags+1)]
            sqRetAuto = autocor(returnsBSESN_Weekly.^2)[1:(maxLags+1)]

            indexPlot = plot(xVals, retAuto, seriestype = :line, title = "BSE Sensex Index",
            xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
            tick_direction = :none, color = "deeppink2", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

            hline!([0], label = false, color =:black, lw = 0.75)

            scatter!(xVals, retAuto, marker = (:circle, 4, stroke(:deeppink2)), 
            color = "deeppink2", label = "Returns")

            plot!(xVals, absRetAuto, seriestype = :line, label = false, 
            color = absCol, lw = 1.5)

            scatter!(xVals, absRetAuto, marker = (:square, 4, stroke(absCol)), 
            color = absCol, label = "|Returns|")

            plot!(xVals, sqRetAuto, seriestype = :line, label = false, 
            color = sqCol, lw = 1.5)

            scatter!(xVals, sqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
            color = sqCol, label = "Returns" * super("2"))

        end

    end

    a1RetAuto = autocor(Returns[t])[1:(maxLags+1)]
    a1AbsRetAuto = (autocor(abs.(Returns[t]))[1:(maxLags+1)])
    a1SqRetAuto = autocor(Returns[t].^2)[1:(maxLags+1)]

    p1 = plot(xVals, a1RetAuto, seriestype = :line, title = "Risky Asset",
    xlabel = "Lag", ylabel = "Autocorrelation", framestyle = :box, 
    tick_direction = :none, color = "darkorange2", lw = 1.5, 
    gridlinewidth = 1.5, gridstyle = :dash, gridalpha = 0.125, label = false)

    scatter!(xVals, a1RetAuto, marker = (:circle, 4, stroke(:darkorange2)), 
    color = "darkorange2", label = "Returns")

    hline!([0], label = false, color =:black, lw = 0.75)

    plot!(xVals, a1AbsRetAuto, seriestype = :line, label = false, 
    color = absCol, lw = 1.5)

    scatter!(xVals, a1AbsRetAuto, marker = (:square, 4, stroke(absCol)), 
    color = absCol, label = "|Returns|")

    plot!(xVals, a1SqRetAuto, seriestype = :line, label = false, 
    color = sqCol, lw = 1.5)

    scatter!(xVals, a1SqRetAuto, marker = (:utriangle, 4, stroke(sqCol)), 
    color = sqCol, label = "Returns" * super("2"))

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotAutoCorrelations(returns, plotStart, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotAutoCorrelations(returns, plotStart, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotAutoCorrelations(returns, plotStart, plotEnd_Daily_BSESN, "BSE", "Daily"))

###############################################################################

