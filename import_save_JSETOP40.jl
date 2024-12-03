### Load JSE TOP40

using XLSX
using JLD2
using DataFrames
using Dates

file_path = "Data/BRICs-EOD-Compact-2005-2015.xlsx"

file = XLSX.openxlsx(file_path)
sheet = file[2]

dates = Date("2005-01-01"):Day(1):Date("2016-04-29")

# Read a specific range of cells (JSE TOP40 Index 2005-2015)
data = sheet["B4:B4141"]
prices = vec(data[2:end, :])

dailyPrices = DataFrame(Date = dates, Price = prices)

# Add ISO week number
dailyPrices.Week = Dates.week.(dailyPrices.Date)

# Add Year
dailyPrices.Year = Dates.year.(dailyPrices.Date)

# Add month
dailyPrices.Month = Dates.month.(dailyPrices.Date)

for row in eachrow(dailyPrices)
    if row.Week == 1 && row.Month == 12
        row.Year += 1
    elseif row.Week >= 52 && row.Month == 1
        row.Year -= 1
    end
end

# Remove rows with missing values
dailyPrices = dropmissing(dailyPrices, :Price)

# Group by year and week, and take the closing price of each week
weeklyPrices = combine(groupby(dailyPrices, [:Year, :Week]), 
:Price => last => :WeeklyClosingPrice)

weekly_JSETOP40_Data = vec(weeklyPrices.WeeklyClosingPrice)
daily_JSETOP40_Data = Float64.(vec(dailyPrices.Price))

# Save daily and weekly price vectors to a .jld2 file
@save "Data/jsetop40_weekly.jld2" weekly_JSETOP40_Data
@save "Data/jsetop40_daily.jld2" daily_JSETOP40_Data

