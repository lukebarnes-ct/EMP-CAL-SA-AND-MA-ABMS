### Load JSE TOP40 demand_Chart

using XLSX
using JLD2

file_path = "Data/BRICs-EOD-Compact-2005-2015.xlsx"

file = XLSX.openxlsx(file_path)
sheet = file[2]

# Read a specific range of cells (JSE TOP40 Index 2005-2015)
data = sheet["B4:B4141"]

cleanData = data[.!any(ismissing, data, dims=2), :]
cleanData = cleanData[2:end, :]

# Currently the data is daily, we transform it into weekly data 
# by simply averaging every set of five Values

weeklyData = [mean(cleanData[i:i+4]) for i in 1:5:length(cleanData) if i+4 <= length(cleanData)]

# Save vector to a .jld2 file
@save "Data/jsetop40.jld2" weeklyData

