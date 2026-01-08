##### Plotting Additional Plots of Interest for Empirical and Simulated Data
##### Lagged Variance, Zunbach Effect and Pareto Distribution over Tail Events

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
@load "Data/hl-calibration/parameters-jse-daily.jld2" optParam_HL_JSE_Daily

@load "Data/hl-calibration/prices-jse-weekly.jld2" prices_HL_JSE_Weekly
@load "Data/hl-calibration/log-returns-jse-weekly.jld2" logReturns_HL_JSE_Weekly
@load "Data/hl-calibration/parameters-jse-weekly.jld2" optParam_HL_JSE_Weekly

# Load ABM Results calibrated to the SSE 50 Index from .jld2 file
@load "Data/hl-calibration/prices-sse50-daily.jld2" prices_HL_SSE50_Daily
@load "Data/hl-calibration/log-returns-sse50-daily.jld2" logReturns_HL_SSE50_Daily
@load "Data/hl-calibration/parameters-sse50-daily.jld2" optParam_HL_SSE50_Daily

@load "Data/hl-calibration/prices-sse50-weekly.jld2" prices_HL_SSE50_Weekly
@load "Data/hl-calibration/log-returns-sse50-weekly.jld2" logReturns_HL_SSE50_Weekly
@load "Data/hl-calibration/parameters-sse50-weekly.jld2" optParam_HL_SSE50_Weekly

# Load ABM Results calibrated to the BSE Sensex Index from .jld2 file
@load "Data/hl-calibration/prices-bsesn-daily.jld2" prices_HL_BSESN_Daily
@load "Data/hl-calibration/log-returns-bsesn-daily.jld2" logReturns_HL_BSESN_Daily
@load "Data/hl-calibration/parameters-bsesn-daily.jld2" optParam_HL_BSESN_Daily

@load "Data/hl-calibration/prices-bsesn-weekly.jld2" prices_HL_BSESN_Weekly
@load "Data/hl-calibration/log-returns-bsesn-weekly.jld2" logReturns_HL_BSESN_Weekly
@load "Data/hl-calibration/parameters-bsesn-weekly.jld2" optParam_HL_BSESN_Weekly

lengthJSE_Daily_HL = length(prices_HL_JSE_Daily)
lengthJSE_Weekly_HL = length(prices_HL_JSE_Weekly)

lengthSSE50_Daily_HL = length(prices_HL_SSE50_Daily)
lengthSSE50_Weekly_HL = length(prices_HL_SSE50_Weekly)

lengthBSESN_Daily_HL = length(prices_HL_BSESN_Daily)
lengthBSESN_Weekly_HL = length(prices_HL_BSESN_Weekly)

#################################################################################

##### Zunbach Effect

maxLags = 25
xVals = 1:maxLags

function zumbach_effect(r::Vector{<:Real},
                        rv::Vector{<:Real},
                        maxlag::Int)

    @assert length(r) == length(rv)
    T = length(r)

    Z = zeros(maxlag)

    for τ in 1:maxlag
        s = 0.0
        for t in 1:(T-τ)
            s += r[t]^2 * rv[t+τ] - r[t+τ]^2 * rv[t]
        end
        Z[τ] = s / (T - τ)
    end

    return Z
end

function rolling_var(r, window)
    [var(r[max(1,i-window+1):i]) for i in eachindex(r)]
end

rv = rolling_var(logReturns_HL_JSE_Daily, 5)  # 5-day variance proxy
Z  = zumbach_effect(logReturns_HL_JSE_Daily[2:end], rv[2:end], maxLags)

plot(xVals, Z, seriestype = :line, label = false, lw = 1.5)

#################################################################################

##### Pareto Distribution/Mean Excess Plots

function mean_excess(x, us)
    me = zeros(length(us))
    for (i, u) in enumerate(us)
        exceed = x[x .> u]
        me[i] = mean(exceed .- u)
    end
    return me
end

# Example:
x_emp = returnsJSE_Daily   # lower tail
x_sim = logReturns_HL_JSE_Daily

us = quantile(x_emp, 0.9:0.005:0.995)

me_emp = mean_excess(x_emp, us)
me_sim = mean_excess(x_sim, us)

plot(us, me_emp, label="Empirical")
plot!(us, me_sim, label="Simulated")