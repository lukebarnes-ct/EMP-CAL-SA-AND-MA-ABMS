
##### XU et al Multi Asset Agent Based Model 

using Random
using Plots
using Distributions
using Printf
using Plots.PlotMeasures
using StatsBase
using TypedTables
using StatsPlots
using Subscripts
using JLD2
using HypothesisTests
using Hurst

#################################################################################

# Load JSE Top 40 Index and Log Returns from .jld2 file
@load "Data/jsetop40_weekly.jld2" weekly_JSETOP40_Data
@load "Data/jsetop40_weekly_LogReturns.jld2" returnsJSE_Weekly

# Load SSE 50 Index and Log Returns from .jld2 file
@load "Data/sse50_weekly.jld2" weekly_SSE50_Data
@load "Data/sse50_weekly_LogReturns.jld2" returnsSSE50_Weekly

# Load BSE Sensex Index and Log Returns from .jld2 file
@load "Data/bsesn_weekly.jld2" weekly_BSESN_Data
@load "Data/bsesn_weekly_LogReturns.jld2" returnsBSESN_Weekly

lengthJSE_Weekly = length(weekly_JSETOP40_Data)

lengthSSE50_Weekly = length(weekly_SSE50_Data)

lengthBSESN_Weekly = length(weekly_BSESN_Data)

#################################################################################

# Load ABM Results calibrated to the JSE Top 40 Index from .jld2 file
@load "Data/xu-calibration/prices-jse-weekly.jld2" priceMatrix_XU_JSE_Weekly
@load "Data/xu-calibration/log-returns-jse-weekly.jld2" logReturnsMatrix_XU_JSE_Weekly
@load "Data/xu-calibration/parameters-jse-weekly.jld2" bestParameters_JSE_Weekly
@load "Data/xu-calibration/objective-results-jse-weekly.jld2" output__JSE_Weekly
@load "Data/xu-calibration/xu-index-price-jse-weekly.jld2" xuIndexPrice_JSE_Weekly
@load "Data/xu-calibration/xu-index-log-returns-jse-weekly.jld2" xuIndexReturn_JSE_Weekly

# Load ABM Results calibrated to the SSE 50 Index from .jld2 file
@load "Data/xu-calibration/prices-sse50-weekly.jld2" priceMatrix_XU_SSE50_Weekly
@load "Data/xu-calibration/log-returns-sse50-weekly.jld2" logReturnsMatrix_XU_SSE50_Weekly
@load "Data/xu-calibration/parameters-sse50-weekly.jld2" bestParameters_SSE50_Weekly
@load "Data/xu-calibration/objective-results-sse50-weekly.jld2" output__SSE50_Weekly
@load "Data/xu-calibration/xu-index-price-sse50-weekly.jld2" xuIndexPrice_SSE50_Weekly
@load "Data/xu-calibration/xu-index-log-returns-sse50-weekly.jld2" xuIndexReturn_SSE50_Weekly

# Load ABM Results calibrated to the BSE Sensex Index from .jld2 file
@load "Data/xu-calibration/prices-bsesn-weekly.jld2" priceMatrix_XU_BSESN_Weekly
@load "Data/xu-calibration/log-returns-bsesn-weekly.jld2" logReturnsMatrix_XU_BSESN_Weekly
@load "Data/xu-calibration/parameters-bsesn-weekly.jld2" bestParameters_BSESN_Weekly
@load "Data/xu-calibration/objective-results-bsesn-weekly.jld2" output__BSESN_Weekly
@load "Data/xu-calibration/xu-index-price-bsesn-weekly.jld2" xuIndexPrice_BSESN_Weekly
@load "Data/xu-calibration/xu-index-log-returns-bsesn-weekly.jld2" xuIndexReturn_BSESN_Weekly

lengthJSE_Weekly_XU = length(xuIndexPrice_JSE_Weekly)

lengthSSE50_Weekly_XU = length(xuIndexPrice_SSE50_Weekly)

lengthBSESN_Weekly_XU = length(xuIndexPrice_BSESN_Weekly)

#################################################################################

plotStart_Weekly = 1
plotEnd_Weekly_JSE = lengthJSE_Weekly_XU
plotEnd_Weekly_SSE50 = lengthSSE50_Weekly_XU
plotEnd_Weekly_BSESN = lengthBSESN_Weekly_XU

###############################################################################

function teststatistic(x)
    n = x.n_x*x.n_y/(x.n_x+x.n_y)
    sqrt(n)*x.δ
end

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

    descStat = Table(Asset = ["HL ABM", "Index", "Difference"], 
    Mean = [abmU[1], indexU[1], abmU[1] - indexU[1]],
    SD = [abmU[2], indexU[2], abmU[2] - indexU[2]],
    Skewness = [abmU[3], indexU[3], abmU[3] - indexU[3]],
    Kurtosis = [abmU[4], indexU[4], abmU[4] - indexU[4]],
    KS_Statistic = [abmU[5], indexU[5], abmU[5] - indexU[5]],
    Hurst = [abmU[6], indexU[6], abmU[6] - indexU[6]])

    return descStat
end

returnStatisticsJSE_Weekly = descriptiveStatistics(xuIndexReturn_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, returnsJSE_Weekly)
returnStatisticsSSE_Weekly = descriptiveStatistics(xuIndexReturn_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, returnsSSE50_Weekly)
returnStatisticsBSE_Weekly = descriptiveStatistics(xuIndexReturn_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, returnsBSESN_Weekly)

###############################################################################

function descriptiveStatisticsWithAssets(AssetReturns, Returns, bt, et, indexReturns)

    t = bt:et

    asset1 = zeros(6)
    asset2 = zeros(6)
    asset3 = zeros(6)
    asset4 = zeros(6)
    asset5 = zeros(6)
    abmU = zeros(6)
    indexU = zeros(6)

    asset1[1] = round(mean(AssetReturns[1, t]), digits = 4)
    asset2[1] = round(mean(AssetReturns[2, t]), digits = 4)
    asset3[1] = round(mean(AssetReturns[3, t]), digits = 4)
    asset4[1] = round(mean(AssetReturns[4, t]), digits = 4)
    asset5[1] = round(mean(AssetReturns[5, t]), digits = 4)

    abmU[1] = round(mean(Returns[t]), digits = 4)
    indexU[1] = round(mean(indexReturns), digits = 4)

    asset1[2] = round(std(AssetReturns[1, t]), digits = 4)
    asset2[2] = round(std(AssetReturns[2, t]), digits = 4)
    asset3[2] = round(std(AssetReturns[3, t]), digits = 4)
    asset4[2] = round(std(AssetReturns[4, t]), digits = 4)
    asset5[2] = round(std(AssetReturns[5, t]), digits = 4)

    abmU[2] = round(std(Returns[t]), digits = 4)
    indexU[2] = round(std(indexReturns), digits = 4)

    asset1[3] = round(skewness(AssetReturns[1, t]), digits = 4)
    asset2[3] = round(skewness(AssetReturns[2, t]), digits = 4)
    asset3[3] = round(skewness(AssetReturns[3, t]), digits = 4)
    asset4[3] = round(skewness(AssetReturns[4, t]), digits = 4)
    asset5[3] = round(skewness(AssetReturns[5, t]), digits = 4)

    abmU[3] = round(skewness(Returns[t]), digits = 4)
    indexU[3] = round(skewness(indexReturns), digits = 4)

    asset1[4] = round(kurtosis(AssetReturns[1, t]), digits = 4)
    asset2[4] = round(kurtosis(AssetReturns[2, t]), digits = 4)
    asset3[4] = round(kurtosis(AssetReturns[3, t]), digits = 4)
    asset4[4] = round(kurtosis(AssetReturns[4, t]), digits = 4)
    asset5[4] = round(kurtosis(AssetReturns[5, t]), digits = 4)

    abmU[4] = round(kurtosis(Returns[t]), digits = 4)
    indexU[4] = round(kurtosis(indexReturns), digits = 4)

    ks_test1 = ApproximateTwoSampleKSTest(AssetReturns[1, t], indexReturns)
    asset1[5] = round(teststatistic(ks_test1), digits = 4)
    ks_test2 = ApproximateTwoSampleKSTest(AssetReturns[2, t], indexReturns)
    asset2[5] = round(teststatistic(ks_test2), digits = 4)
    ks_test3 = ApproximateTwoSampleKSTest(AssetReturns[3, t], indexReturns)
    asset3[5] = round(teststatistic(ks_test3), digits = 4)
    ks_test4 = ApproximateTwoSampleKSTest(AssetReturns[4, t], indexReturns)
    asset4[5] = round(teststatistic(ks_test4), digits = 4)
    ks_test5 = ApproximateTwoSampleKSTest(AssetReturns[5, t], indexReturns)
    asset5[5] = round(teststatistic(ks_test5), digits = 4)

    ks_test = ApproximateTwoSampleKSTest(Returns[t], indexReturns)
    abmU[5] = round(teststatistic(ks_test), digits = 4)
    indexU[5] = 0.0000

    asset1[6] = round(hurst_exponent(AssetReturns[1, t], 1:19)[1], digits = 4)
    asset2[6] = round(hurst_exponent(AssetReturns[2, t], 1:19)[1], digits = 4)
    asset3[6] = round(hurst_exponent(AssetReturns[3, t], 1:19)[1], digits = 4)
    asset4[6] = round(hurst_exponent(AssetReturns[4, t], 1:19)[1], digits = 4)
    asset5[6] = round(hurst_exponent(AssetReturns[5, t], 1:19)[1], digits = 4)

    abmU[6] = round(hurst_exponent(Returns[t], 1:19)[1], digits = 4)
    indexU[6] = round(hurst_exponent(indexReturns, 1:19)[1], digits = 4)

    descStat = Table(Asset = ["A1", "A2", "A3", "A4", "A5", "XU Index", "Index", "Difference"], 
    Mean = [asset1[1], asset2[1], asset3[1], asset4[1], asset5[1], abmU[1], indexU[1], abmU[1] - indexU[1]],
    SD = [asset1[2], asset2[2], asset3[2], asset4[2], asset5[2], abmU[2], indexU[2], abmU[2] - indexU[2]],
    Skewness = [asset1[3], asset2[3], asset3[3], asset4[3], asset5[3], abmU[3], indexU[3], abmU[3] - indexU[3]],
    Kurtosis = [asset1[4], asset2[4], asset3[4], asset4[4], asset5[4], abmU[4], indexU[4], abmU[4] - indexU[4]],
    KS_Statistic = [asset1[5], asset2[5], asset3[5], asset4[5], asset5[5], abmU[5], indexU[5], abmU[5] - indexU[5]],
    Hurst = [asset1[6], asset2[6], asset3[6], asset4[6], asset5[6], abmU[6], indexU[6], abmU[6] - indexU[6]])

    return descStat
end

returnStatisticsJSE_Weekly_ALL = descriptiveStatisticsWithAssets(logReturnsMatrix_XU_JSE_Weekly[:, 101:691], xuIndexReturn_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, returnsJSE_Weekly)
returnStatisticsSSE_Weekly_ALL = descriptiveStatisticsWithAssets(logReturnsMatrix_XU_SSE50_Weekly[:, 101:679], xuIndexReturn_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, returnsSSE50_Weekly)
returnStatisticsBSE_Weekly_ALL = descriptiveStatisticsWithAssets(logReturnsMatrix_XU_BSESN_Weekly[:, 101:690], xuIndexReturn_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, returnsBSESN_Weekly)

###############################################################################

# Price Plots

function plotPrices(Prices, bt, et, index)

    t = bt:et

    if index == "JSE"

        indexPlot = plot(1:lengthJSE_Weekly, weekly_JSETOP40_Data, label = "JSE Top 40", title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "purple1", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

    elseif index == "SSE"

        indexPlot = plot(1:lengthSSE50_Weekly, weekly_SSE50_Data, label = "SSE 50", title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "dodgerblue4", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)

    elseif index == "BSE"

        indexPlot = plot(1:lengthBSESN_Weekly, weekly_BSESN_Data, label = "BSE Sensex", title = "BSE Sensex Index", 
               xlabel = "Week", ylabel = "Price", legend = false, 
               yformatter = x -> @sprintf("%.0f", x), framestyle = :box, 
               tick_direction = :none, color = "deeppink2", lw = 1.5, 
               gridlinewidth = 1.5, gridstyle = :dash)
    end

    p1 = plot(t, Prices[t], label = "Price", title = "Simulated Index", 
        xlabel = "Week", ylabel = "Price", legend = false, framestyle = :box, 
        tick_direction = :none, color = "darkorange2", lw = 1.5, 
        gridlinewidth = 1.5, gridstyle = :dash)

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
         margin = 2mm)
end

display(plotPrices(xuIndexPrice_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE"))
savefig("Plots/xu-calibration/prices/jse_weekly.pdf")
display(plotPrices(xuIndexPrice_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE"))
savefig("Plots/xu-calibration/prices/sse50_weekly.pdf")
display(plotPrices(xuIndexPrice_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE"))
savefig("Plots/xu-calibration/prices/bsesn_weekly.pdf")

###############################################################################

# Asset Return Plots

function plotReturns(Returns, bt, et, index)

    t = bt:et

    if index == "JSE"

        indexPlot = plot(1:lengthJSE_Weekly, returnsJSE_Weekly, label = false, title = "JSE Top 40 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "purple1", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsJSE_Weekly)], label = round(mean(returnsJSE_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    elseif index == "SSE"

        indexPlot = plot(1:lengthSSE50_Weekly, returnsSSE50_Weekly, label = false, title = "SSE 50 Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "dodgerblue4", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsSSE50_Weekly)], label = round(mean(returnsSSE50_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    elseif index == "BSE"

        indexPlot = plot(1:lengthBSESN_Weekly, returnsBSESN_Weekly, label = false, title = "BSE Sensex Index", 
               xlabel = "Week", ylabel = "Return", legend = :topleft, 
               framestyle = :box, tick_direction = :none, color = "deeppink2", 
               ylim = (-0.15, 0.25), grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

               hline!([mean(returnsBSESN_Weekly)], label = round(mean(returnsBSESN_Weekly), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    end

    p1 = plot(t, Returns[t], label = false, title = "Simulated Index", 
              xlabel = "Week", ylabel = "Returns", legend = :topleft, framestyle = :box, 
              tick_direction = :none, color = "darkorange2", ylim = (-0.15, 0.25), 
              grid = (:y, :auto), gridlinewidth = 1.5, gridalpha = 0.125)

         hline!([mean(Returns[t])], label = round(mean(Returns[t]), digits = 4), 
               color =:black, lw = 1, linestyle =:dash)

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturns(xuIndexReturn_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE"))
savefig("Plots/xu-calibration/returns/jse_weekly.pdf")
display(plotReturns(xuIndexReturn_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE"))
savefig("Plots/xu-calibration/returns/sse50_weekly.pdf")
display(plotReturns(xuIndexReturn_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE"))
savefig("Plots/xu-calibration/returns/bsesn_weekly.pdf")

###############################################################################

# Return Distribution Histogram and Density Plots

function plotReturnDistribution(Returns, bt, et, index)

    t = bt:et
    xVals = range(-0.15, stop = 0.15, length = 60)
    binSize = xVals

    if index == "JSE"

        indexPlot = histogram(returnsJSE_Weekly, bins = binSize, title = "JSE Top 40 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "purple1", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

        densityPlot = pdf.(Normal(mean(returnsJSE_Weekly), std(returnsJSE_Weekly)), xVals)

        plot!(xVals, densityPlot, label = false, lw = 2.5, color = "purple1")

    elseif index == "SSE"

        indexPlot = histogram(returnsSSE50_Weekly, bins = binSize, title = "SSE 50 Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "dodgerblue4", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

        densityPlot = pdf.(Normal(mean(returnsSSE50_Weekly), std(returnsSSE50_Weekly)), xVals)

        plot!(xVals, densityPlot, label = false, lw = 2.5, color = "dodgerblue4")

    elseif index == "BSE"

        indexPlot = histogram(returnsBSESN_Weekly, bins = binSize, title = "BSE Sensex Index",
            xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
            label = false, color = "deeppink2", alpha = 0.5, framestyle = :box, 
            tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
            gridalpha = 0.125)

        densityPlot = pdf.(Normal(mean(returnsBSESN_Weekly), std(returnsBSESN_Weekly)), xVals)

        plot!(xVals, densityPlot, label = false, lw = 2.5, color = "deeppink2")

    end

    p1 = histogram(Returns[t], bins = binSize, title = "Simulated Index", 
    xlabel = "Returns", ylabel = "Density", xlim = (-0.15, 0.15), normalize = :pdf,
    label = false, color = "darkorange2", alpha = 0.5, framestyle = :box, 
    tick_direction = :none, grid = (:y, :auto), gridlinewidth = 1.5, 
    gridalpha = 0.125)

    a1Density = pdf.(Normal(mean(Returns[t]), std(Returns[t])), xVals)
    plot!(xVals, a1Density, label = false, lw = 2.5, color = "darkorange2")

    plot(p1, indexPlot, layout = (2, 1), size = (900, 900), 
    margin = 2mm)

end

display(plotReturnDistribution(xuIndexReturn_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE"))
savefig("Plots/xu-calibration/return-distribution/jse_weekly.pdf")
display(plotReturnDistribution(xuIndexReturn_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE"))
savefig("Plots/xu-calibration/return-distribution/sse50_weekly.pdf")
display(plotReturnDistribution(xuIndexReturn_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE"))
savefig("Plots/xu-calibration/return-distribution/bsesn_weekly.pdf")

###############################################################################

# Return AutoCorrelation Plots

function plotAutoCorrelations(Returns, bt, et, index)

    t = bt:et
    maxLags = 25
    xVals = 0:maxLags

    sqCol = "green4"
    absCol = "firebrick"

    if index == "JSE"

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


    elseif index == "SSE"

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

    elseif index == "BSE"

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

display(plotAutoCorrelations(xuIndexReturn_JSE_Weekly, plotStart_Weekly, plotEnd_Weekly_JSE, "JSE"))
savefig("Plots/xu-calibration/autocorrelations/jse_weekly.pdf")
display(plotAutoCorrelations(xuIndexReturn_SSE50_Weekly, plotStart_Weekly, plotEnd_Weekly_SSE50, "SSE"))
savefig("Plots/xu-calibration/autocorrelations/sse50_weekly.pdf")
display(plotAutoCorrelations(xuIndexReturn_BSESN_Weekly, plotStart_Weekly, plotEnd_Weekly_BSESN, "BSE"))
savefig("Plots/xu-calibration/autocorrelations/bsesn_weekly.pdf")

###############################################################################
