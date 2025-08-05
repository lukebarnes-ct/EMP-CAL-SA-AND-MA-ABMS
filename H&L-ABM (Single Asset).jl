
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

###############################################################################

# Review Output from model and FINAL Parameters

timeEnd = 10000

plotStart_Daily = 1001
plotEnd_Daily_JSE = plotStart_Daily - 1 + lengthJSE_Daily 
plotEnd_Daily_SSE50 = plotStart_Daily - 1 + lengthSSE50_Daily
plotEnd_Daily_BSESN = plotStart_Daily - 1 + lengthBSESN_Daily

plotStart_Weekly = 1001
plotEnd_Weekly_JSE = plotStart_Weekly - 1 + lengthJSE_Weekly 
plotEnd_Weekly_SSE50 = plotStart_Weekly - 1 + lengthSSE50_Weekly
plotEnd_Weekly_BSESN = plotStart_Weekly - 1 + lengthBSESN_Weekly

### Hyperparameters

# Market Maker Price Adjustment Speed
# MU = 2

# Chartist Extrapolation Rate
# GAMMA = 0.3

# Delta
# DELTA = 0.85

# Fundamentalist Price Adjustment Speed
# ALPHA = 0.1

par = [2.1652601173749995, 0.14593493917215378, 1.0, 0.9372527766928516]
par = [7.940640235139954, 9.583791601718158, 0.9725714966050134, 0.5047848572913415]
par = [9.857206413616172, 9.901044140605642, 0.97460808798178, 0.4976363011901914]

MU = par[1]
GAMMA = par[2]
DELTA = par[3] 
ALPHA = par[4]

prices, fv, returns, demFund, demChart, expFund, expChart, exG = hlABM(timeEnd, 20, 20, MU, GAMMA, DELTA, ALPHA)

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

function plotPrices(Prices, FValue, bt, et, index, timescale, plotFV)

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

    if timescale == "Daily"

        p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset", 
        xlabel = "Day", ylabel = "Price", legend = false, framestyle = :box, 
        tick_direction = :none, color = "darkorange2", lw = 1.5, 
        gridlinewidth = 1.5, gridstyle = :dash)

        if plotFV == true

            pf =  plot(t, FValue[t], label = "Fundamental Value",
            xlabel = "Day", ylabel = "FV", legend = false, framestyle = :box, 
            tick_direction = :none, color = "red", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash)

            plot(p1, pf, indexPlot, layout = (3, 1), size = (900, 900), 
            margin = 2mm)

        else

            plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
            margin = 2mm)

        end

    elseif timescale == "Weekly"

        p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset", 
        xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
        tick_direction = :none, color = "darkorange2", lw = 1.5, 
        gridlinewidth = 1.5, gridstyle = :dash)

        if plotFV == true

            pf =  plot(t, FValue[t], label = "Fundamental Value",
            xlabel = "Week", ylabel = "FV", legend = false, framestyle = :box, 
            tick_direction = :none, color = "red", lw = 1.5, 
            gridlinewidth = 1.5, gridstyle = :dash)

            plot(p1, pf, indexPlot, layout = (3, 1), size = (900, 900), 
            margin = 2mm)

        else

            plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
            margin = 2mm)

        end

    end

end

# With Fundamental Value

display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily", true))
display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily", true))
display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily", true))

# Without Fundamental Value

display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily", false))
display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily", false))
display(plotPrices(prices, fv, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily", false))

###############################################################################

# Asset Return Plots

function plotReturns(Returns, bt, et, index, timescale)

    t = bt:et

    if index == "JSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthJSE_Daily, returnsJSE_Daily, label = false, title = "JSE Top 40 Index", 
            xlabel = "Day", ylabel = "Return", legend = :topleft, 
            framestyle = :box, tick_direction = :none, color = "purple1", 
            ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

            hline!([mean(returnsJSE_Daily)], label = round(mean(returnsJSE_Daily), digits = 4), 
            color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthJSE_Weekly, returnsJSE_Weekly, label = false, title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "purple1", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsJSE_Weekly)], label = round(mean(returnsJSE_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        end

    elseif index == "SSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthSSE50_Daily, returnsSSE50_Daily, label = false, title = "SSE 50 Index", 
               xlabel = "Day", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "dodgerblue4", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsSSE50_Daily)], label = round(mean(returnsSSE50_Daily), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthSSE50_Weekly, returnsSSE50_Weekly, label = false, title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "dodgerblue4", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsSSE50_Weekly)], label = round(mean(returnsSSE50_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        end

    elseif index == "BSE"

        if timescale == "Daily"

            indexPlot = plot(1:lengthBSESN_Daily, returnsBSESN_Daily, label = false, title = "BSE Sensex Index", 
               xlabel = "Day", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "deeppink2", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsBSESN_Daily)], label = round(mean(returnsBSESN_Daily), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

        elseif timescale == "Weekly"

            indexPlot = plot(1:lengthBSESN_Weekly, returnsBSESN_Weekly, label = false, title = "BSE Sensex Index", 
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

display(plotReturns(returns, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotReturns(returns, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotReturns(returns, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))

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

display(plotReturnDistribution(returns, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotReturnDistribution(returns, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotReturnDistribution(returns, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))

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

display(plotAutoCorrelations(returns, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
display(plotAutoCorrelations(returns, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
display(plotAutoCorrelations(returns, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))

###############################################################################

###############################################################################

tt = 1:10
bt = 9201
et = 9765
prices[tt]

fv[tt]
returns[tt]
demFund[tt]

function printOutput(bt, et, type)

    head = ["$bt", "$bt+1", "$bt+2", "$bt+3", "$bt+4", "$et-4", "$et-3", "$et-2", "$et-1", "$et"]

    lt = length(bt:et)

    if type == "Demand"
        println("Fundamentalists Demand")
        pretty_table(transpose(demFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Demand")
        pretty_table(transpose(demChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)

    elseif type == "ER"
        println("Fundamentalists Expected Return")
        pretty_table(transpose(expFund[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
                    header = head)
        println("Chartists Expected Return")
        pretty_table(transpose(expChart[[bt, bt+1, bt+2, bt+3, bt+4, et-4, et-3, et-2, et-1, et]]),
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

printOutput(bt, et, "Demand")
printOutput(bt, et, "ER")
printOutput(bt, et, "Price")

printOutput(plotStart, plotEnd_Daily_JSE, "ER")
printOutput(plotStart, plotEnd_Daily_JSE, "Demand")
printOutput(plotStart, plotEnd_Daily_JSE, "Price")

printOutput(plotStart, plotEnd_Daily_SSE50, "ER")
printOutput(plotStart, plotEnd_Daily_SSE50, "Demand")
printOutput(plotStart, plotEnd_Daily_SSE50, "Price")

printOutput(plotStart, plotEnd_Daily_BSESN, "ER")
printOutput(plotStart, plotEnd_Daily_BSESN, "Demand")
printOutput(plotStart, plotEnd_Daily_BSESN, "Price")
