
using JLD2

# Load JSE Top 40 Index from .jld2 file
@load "Data/jsetop40_weekly.jld2" weekly_JSETOP40_Data
@load "Data/jsetop40_daily.jld2" daily_JSETOP40_Data

# Load SSE 50 Index from .jld2 file
@load "Data/sse50_weekly.jld2" weekly_SSE50_Data
@load "Data/sse50_daily.jld2" daily_SSE50_Data

# Load BSE Sensex Index from .jld2 file
@load "Data/bsesn_weekly.jld2" weekly_BSESN_Data
@load "Data/bsesn_daily.jld2" daily_BSESN_Data

# Calculate the Returns/Log Returns of each Index

function calculateReturns(p, type)
    
    len = length(p)

    if type == "Simple"

        r = zeros(len)

        for i in 2:len

            r[i] = (p[i] - p[i-1]) / p[i-1]
        end

    elseif type == "Log"

        r = zeros(len)

        for i in 2:len

            r[i] = log(p[i] / p[i-1])
        end
    end

    return r

end

#################################################################################

# Calculate the Daily and Weekly Log Returns of the 
# JSE Top 40 Price Data, 
# SSE50 Price Data and 
# BSESN Price Data

returnsJSE_Daily = calculateReturns(daily_JSETOP40_Data, "Log")
returnsJSE_Weekly = calculateReturns(weekly_JSETOP40_Data, "Log")

returnsSSE50_Daily = calculateReturns(daily_SSE50_Data, "Log")
returnsSSE50_Weekly = calculateReturns(weekly_SSE50_Data, "Log")

returnsBSESN_Daily = calculateReturns(daily_BSESN_Data, "Log")
returnsBSESN_Weekly = calculateReturns(weekly_BSESN_Data, "Log")

# Save Daily and Weekly log return vectors to a .jld2 file
@save "Data/jsetop40_weekly_LogReturns.jld2" returnsJSE_Weekly
@save "Data/jsetop40_daily_LogReturns.jld2" returnsJSE_Daily

@save "Data/sse50_weekly_LogReturns.jld2" returnsSSE50_Weekly
@save "Data/sse50_daily_LogReturns.jld2" returnsSSE50_Daily

@save "Data/bsesn_weekly_LogReturns.jld2" returnsBSESN_Weekly
@save "Data/bsesn_daily_LogReturns.jld2" returnsBSESN_Daily

#################################################################################

