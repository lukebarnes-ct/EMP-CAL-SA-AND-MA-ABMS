##### He and Li 2008 Agent Based Model 

using Random
using Plots
using Distributions
using Printf
using Plots.PlotMeasures
using StatsBase
using TypedTables
using StatsPlots
using Subscripts

#################################################################################

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

# Load ABM Results calibrated to the JSE Top 40 Index from .jld2 file
@load "Data/hl-calibration/prices-jse-daily.jld2" prices_HL_JSE_Daily
@load "Data/hl-calibration/log-returns-jse-daily.jld2" logReturns_HL_JSE_Daily
@load "Data/hl-calibration/prices-jse-weekly.jld2" prices_HL_JSE_Weekly
@load "Data/hl-calibration/log-returns-jse-weekly.jld2" logReturns_HL_JSE_Weekly

# Load ABM Results calibrated to the SSE 50 Index from .jld2 file
@load "Data/hl-calibration/prices-sse50-daily.jld2" prices_HL_SSE50_Daily
@load "Data/hl-calibration/log-returns-sse50-daily.jld2" logReturns_HL_SSE50_Daily
@load "Data/hl-calibration/prices-sse50-weekly.jld2" prices_HL_SSE50_Weekly
@load "Data/hl-calibration/log-returns-sse50-weekly.jld2" logReturns_HL_SSE50_Weekly

# Load ABM Results calibrated to the BSE Sensex Index from .jld2 file
@load "Data/hl-calibration/prices-bsesn-daily.jld2" prices_HL_BSESN_Daily
@load "Data/hl-calibration/log-returns-bsesn-daily.jld2" logReturns_HL_BSESN_Daily
@load "Data/hl-calibration/prices-bsesn-weekly.jld2" prices_HL_BSESN_Weekly
@load "Data/hl-calibration/log-returns-bsesn-weekly.jld2" logReturns_HL_BSESN_Weekly

lengthJSE_Daily_HL = length(prices_HL_JSE_Daily)
lengthJSE_Weekly_HL = length(prices_HL_JSE_Weekly)

lengthSSE50_Daily_HL = length(prices_HL_SSE50_Daily)
lengthSSE50_Weekly_HL = length(prices_HL_SSE50_Weekly)

lengthBSESN_Daily_HL = length(prices_HL_BSESN_Daily)
lengthBSESN_Weekly_HL = length(prices_HL_BSESN_Weekly)

#################################################################################

plotStart_Daily = 1001
plotEnd_Daily_JSE = plotStart_Daily - 1 + lengthJSE_Daily 
plotEnd_Daily_SSE50 = plotStart_Daily - 1 + lengthSSE50_Daily
plotEnd_Daily_BSESN = plotStart_Daily - 1 + lengthBSESN_Daily

plotStart_Weekly = 1001
plotEnd_Weekly_JSE = plotStart_Weekly - 1 + lengthJSE_Weekly 
plotEnd_Weekly_SSE50 = plotStart_Weekly - 1 + lengthSSE50_Weekly
plotEnd_Weekly_BSESN = plotStart_Weekly - 1 + lengthBSESN_Weekly

###############################################################################

# Descriptive Statistics of Asset Returns

function descriptiveStatistics(Returns, bt, et, indexReturns)

    t = bt:et

    abmU = zeros(6)
    indexU = zeros(6)

    abmU[1] = round(mean(Returns[t]), digits = 4)
    indexU[1] = round(mean(indexReturns), digits = 4)

    abmU[2] = round(std(Returns[t]), digits = 4)
    indexU[2] = round(std(indexReturns), digits = 4)

    abmU[3] = round(skewness(Returns[t]), digits = 4)
    indexU[3] = round(skewness(indexReturns), digits = 4)

    abmU[4] = round(kurtosis(Returns[t]), digits = 4)
    indexU[4] = round(kurtosis(indexReturns), digits = 4)

    ks_test = ApproximateTwoSampleKSTest(Returns[t], indexReturns)
    abmU[5] = round(teststatistic(ks_test), digits = 4)
    indexU[5] = 0.0000

    abmU[6] = round(hurst_exponent(Returns[t], 1:19)[1], digits = 4)
    indexU[6] = round(hurst_exponent(indexReturns, 1:19)[1], digits = 4)

    descStat = Table(Asset = ["HL ABM", "Index"], 
    Mean = [abmU[1], indexU[1]],
    SD = [abmU[2], indexU[2]],
    Skewness = [abmU[3], indexU[3]],
    Kurtosis = [abmU[4], indexU[4]])
    KS_Statistic = [abmU[5], indexU[5]],
    Hurst = ([abmU[6], indexU[6]])

    return descStat
end

returnStatisticsJSE_Daily = descriptiveStatistics(logReturns_HL_JSE_Daily, plotStart_Daily, plotEnd_Daily_JSE, returnsJSE_Daily)
returnStatisticsSSE_Daily = descriptiveStatistics(logReturns_HL_SSE50_Daily, plotStart_Daily, plotEnd_Daily_SSE50, returnsSSE50_Daily)
returnStatisticsBSE_Daily = descriptiveStatistics(logReturns_HL_BSESN_Daily, plotStart_Daily, plotEnd_Daily_BSESN, returnsBSESN_Daily)

returnStatisticsJSE_Weekly = descriptiveStatistics(logReturns_HL_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, returnsJSE_Weekly)
returnStatisticsSSE_Weekly = descriptiveStatistics(logReturns_HL_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, returnsSSE50_Weekly)
returnStatisticsBSE_Weekly = descriptiveStatistics(logReturns_HL_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, returnsBSESN_Weekly)

###############################################################################

# Price Plots

function plotPrices(Prices, bt, et, index, timescale)

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

    elseif timescale == "Weekly"

        p1 = plot(t, Prices[t], label = "Price", title = "Risky Asset", 
        xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
        tick_direction = :none, color = "darkorange2", lw = 1.5, 
        gridlinewidth = 1.5, gridstyle = :dash)

    end

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
         margin = 2mm)
end

display(plotPrices(prices_HL_JSE_Daily, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
savefig("Plots/hl-calibration/prices/jse_daily.pdf")
display(plotPrices(prices_HL_SSE50_Daily, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
savefig("Plots/hl-calibration/prices/sse50_daily.pdf")
display(plotPrices(prices_HL_BSESN_Daily, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))
savefig("Plots/hl-calibration/prices/bsesn_daily.pdf")

display(plotPrices(prices_HL_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE", "Weekly"))
savefig("Plots/hl-calibration/prices/jse_weekly.pdf")
display(plotPrices(prices_HL_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE", "Weekly"))
savefig("Plots/hl-calibration/prices/sse50_weekly.pdf")
display(plotPrices(prices_HL_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE", "Weekly"))
savefig("Plots/hl-calibration/prices/bsesn_weekly.pdf")

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

    if timescale == "Daily"

        p1 = plot(t, Returns[t], label = false, title = "Risky Asset", 
              xlabel = "Day", ylabel = "Returns", legend = :topleft, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
              grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

        hline!([mean(Returns[t])], label = round(mean(Returns[t]), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    elseif timescale == "Weekly"

        p1 = plot(t, Returns[t], label = false, title = "Risky Asset", 
              xlabel = "Week", ylabel = "Returns", legend = :topleft, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
              grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

        hline!([mean(Returns[t])], label = round(mean(Returns[t]), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    end

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturns(logReturns_HL_JSE_Daily, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
savefig("Plots/hl-calibration/returns/jse_daily.pdf")
display(plotReturns(logReturns_HL_SSE50_Daily, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
savefig("Plots/hl-calibration/returns/sse50_daily.pdf")
display(plotReturns(logReturns_HL_BSESN_Daily, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))
savefig("Plots/hl-calibration/returns/bsesn_daily.pdf")

display(plotReturns(logReturns_HL_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE", "Weekly"))
savefig("Plots/hl-calibration/returns/jse_weekly.pdf")
display(plotReturns(logReturns_HL_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE", "Weekly"))
savefig("Plots/hl-calibration/returns/sse50_weekly.pdf")
display(plotReturns(logReturns_HL_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE", "Weekly"))
savefig("Plots/hl-calibration/returns/bsesn_weekly.pdf")

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

display(plotReturnDistribution(logReturns_HL_JSE_Daily, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
savefig("Plots/hl-calibration/return-distribution/jse_daily.pdf")
display(plotReturnDistribution(logReturns_HL_SSE50_Daily, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
savefig("Plots/hl-calibration/return-distribution/sse50_daily.pdf")
display(plotReturnDistribution(logReturns_HL_BSESN_Daily, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))
savefig("Plots/hl-calibration/return-distribution/bsesn_daily.pdf")

display(plotReturnDistribution(logReturns_HL_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE", "Weekly"))
savefig("Plots/hl-calibration/return-distribution/jse_weekly.pdf")
display(plotReturnDistribution(logReturns_HL_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE", "Weekly"))
savefig("Plots/hl-calibration/return-distribution/sse50_weekly.pdf")
display(plotReturnDistribution(logReturns_HL_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE", "Weekly"))
savefig("Plots/hl-calibration/return-distribution/bsesn_weekly.pdf")

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

display(plotAutoCorrelations(logReturns_HL_JSE_Daily, plotStart_Daily, plotEnd_Daily_JSE, "JSE", "Daily"))
savefig("Plots/hl-calibration/autocorrelations/jse_daily.pdf")
display(plotAutoCorrelations(logReturns_HL_SSE50_Daily, plotStart_Daily, plotEnd_Daily_SSE50, "SSE", "Daily"))
savefig("Plots/hl-calibration/autocorrelations/sse50_daily.pdf")
display(plotAutoCorrelations(logReturns_HL_BSESN_Daily, plotStart_Daily, plotEnd_Daily_BSESN, "BSE", "Daily"))
savefig("Plots/hl-calibration/autocorrelations/bsesn_daily.pdf")

display(plotAutoCorrelations(logReturns_HL_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE", "Weekly"))
savefig("Plots/hl-calibration/autocorrelations/jse_weekly.pdf")
display(plotAutoCorrelations(logReturns_HL_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE", "Weekly"))
savefig("Plots/hl-calibration/autocorrelations/sse50_weekly.pdf")
display(plotAutoCorrelations(logReturns_HL_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE", "Weekly"))
savefig("Plots/hl-calibration/autocorrelations/bsesn_weekly.pdf")

###############################################################################
