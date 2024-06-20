##### Xu et al., 2014 Agent Based Model 

using Random
using Plots
using Distributions

### Parameters

T = 1000            # Number of Timesteps
N = 3               # Number of Risky Assets
kChart = 20         # Number of Chartists
kFund = 20          # Number of Fundamentalists

phi = 0.001         # Dividend Growth Rate
phi_sd = 0.01       # Dividend Growth Rate Standard Deviation
r = 0.0012          # Risk Free Rate
lambda = 3          # Relative Risk Aversion

wind_max = 80       # Max Exponential Moving Average Periods
wind_min = 20       # Min Exponential Moving Average Periods

meanR_max = 1       # Max Mean Reversion
meanR_min = 0.5     # Min Mean Reversion

corr_max = 0.8      # Max Expected Correlation Coefficient
corr_min = -0.2     # Min Expected Correlation Coefficient

propW_max = 0.95    # Max Wealth Investment Proportion
propW_min = -0.95   # Min Wealth Investment Proportion 


### Initialise Variables

cash_0 = 10         # Initial Cash 
div_0 = 0.002       # Initial Dividend

dividends = zeros(N, T)         # Dividends of Risky Assets
fund_val = zeros(N, T)          # Fundamental Values of Risky Assets
price = zeros(N, T)             # Prices of Risky Assets

expRet_Fund = zeros(N, T, kFund)                    # Fundamentalists Expected Return of Risky Assets
expRet_Chart = zeros(N, T, kChart)                  # Chartists Expected Return of Risky Assets
expRet_CovMat_Fund = zeros(N, N, T, kFund)          # Expected Return Covariance Array for Fundamentalists
expRet_CovMat_Chart = zeros(N, N, T, kChart)        # Expected Return Covariance Array for Chartists

for i in 1:N
    dividends[i, 1] = div_0       # Set Initial Dividend in Matrix
end

meanR = round.(rand(Uniform(meanR_min, meanR_max), kFund), digits = 2)
ema_wind_Fund = rand(wind_min:wind_max, kFund)
ema_wind_Chart = rand(wind_min:wind_max, kChart)


for t in 2:T

    covVec_Fund = zeros(N, kFund)
    covVec_Chart = zeros(N, kChart)

    for i in 1:N

        err = rand(Normal(0, 1))                                                # Standard Normal Error Term
        dividends[i, t] = (1 + phi + phi_sd * err) * dividends[i, t-1]          # Expected Dividends for Next Time Period
        fund_val[i, t] = (1 + phi + phi_sd * err) * fund_val[i, t-1]            # Expected Fundamental Value for Next Time Period

        for f in 1:kFund

            expRet_Fund[i, t, f] = (phi * fund_val[i, t-1] + meanR[f] * (fund_val[i, t-1] - price[i, t-1]) + (1 + phi) * dividends[i, t-1] - price[i, t-1]) / price[i, t-1]

            ema_f = exp(-1/ema_wind_Fund[f])
            expRet_CovMat_Fund[i, i, t, f] = (ema_f * expRet_CovMat_Fund[i, i, t-1, f]) + (1 - ema_f) * (expRet_Fund[i, t-1, f] - ema_f)
            

        end
        
        for c in 1:kChart


        end

    end
    
end