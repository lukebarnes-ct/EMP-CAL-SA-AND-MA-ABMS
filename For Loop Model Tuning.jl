##################################################################################

### Some Loops

fValues = [4, 8, 20, 40, 100, 200]

for a in fValues

    prices, returns, fundValue, pRet, erFund, erChart, wpFund, wpFund_rf, wpChart, wpChart_rf, 
    wInvFund, wInvFund_rf, wInvChart, wInvChart_rf, wFund, wChart, 
    demFund, demChart, excDem = modelHyperparameters(timeEnd, n, numChart, numFund, 
                                                     wMax, wMin, mRMax, mRMin, 
                                                     corrMax, corrMin, pWMax, pWMin, 
                                                     stockMax, stockMin, a)

    display(plotReturns(returns, BT, ET, numFund, numChart))
    display(plotPrices(prices, fundValue, BT, ET, numFund, numChart))
        
end

####################################################################################

### Some Loops

mValues = [1, 5, 10, 20, 30, 50]

for a in mValues

    for b in mValues

        prices, returns, fundValue, pRet, erFund, erChart, wpFund, wpFund_rf, wpChart, wpChart_rf, 
        wInvFund, wInvFund_rf, wInvChart, wInvChart_rf, wFund, wChart, 
        demFund, demChart, excDem = modelHyperparameters(timeEnd, n, numChart, numFund, 
                                                        wMax, wMin, mRMax, mRMin, 
                                                        corrMax, corrMin, pWMax, pWMin, 
                                                        stockMax, stockMin, fundamental_value,
                                                        a, b)

        display(plotReturns(returns, BT, ET, numFund, numChart))

        function plotPrices(Prices, FValue, bt, et, kF, kC)

            t = bt:et
        
            sz = 250 * n
        
            if n == 2
        
                p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
                      Gamma = [$mRMin, $mRMax], Rho = [$corrMin, $corrMax], 
                      Tau = [$pWMin, $pWMax], Stock = [$stockMin, $stockMax],
                      EMA = [$wMin, $wMax]", 
                      xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[1, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[2, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                plot(p1, p2, layout = (n, 1), size = (800, sz))
        
            elseif n == 3
        
                p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
                      Gamma = [$mRMin, $mRMax], Rho = [$corrMin, $corrMax], 
                      Tau = [$pWMin, $pWMax], Stock = [$stockMin, $stockMax],
                      EMA = [$wMin, $wMax]", 
                      xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[1, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[2, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3, FMult = $a, CMult = $b", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[3, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                plot(p1, p2, p3, layout = (3, 1), size = (800, sz))
        
            elseif n == 4
        
                p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
                      Gamma = [$mRMin, $mRMax], Rho = [$corrMin, $corrMax], 
                      Tau = [$pWMin, $pWMax], Stock = [$stockMin, $stockMax],
                      EMA = [$wMin, $wMax]", 
                      xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[1, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[2, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[3, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p4 = plot(t, Prices[4, t], label = "Price", title = "Asset 4", 
                    xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[4, t], 
                      label = "Fundamental Value", linecolor=:red)
        
                plot(p1, p2, p3, p4, layout = (n, 1), size = (800, sz))
        
            elseif n == 5
        
                p1 = plot(t, Prices[1, t], label = "Price", title = "Asset 1, KF = $kF, KC = $kC, 
                      Gamma = [$mRMin, $mRMax], Rho = [$corrMin, $corrMax], 
                      Tau = [$pWMin, $pWMax], Stock = [$stockMin, $stockMax],
                      EMA = [$wMin, $wMax]", 
                      xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[1, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p2 = plot(t, Prices[2, t], label = "Price", title = "Asset 2", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[2, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p3 = plot(t, Prices[3, t], label = "Price", title = "Asset 3", 
                        xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[3, t], 
                    label = "Fundamental Value", linecolor=:red)
        
                p4 = plot(t, Prices[4, t], label = "Price", title = "Asset 4", 
                    xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[4, t], 
                      label = "Fundamental Value", linecolor=:red)
        
                p5 = plot(t, Prices[5, t], label = "Price", title = "Asset 5", 
                    xlabel = "T", ylabel = "Price", legend = :topleft)
        
                plot!(t, FValue[5, t], 
                      label = "Fundamental Value", linecolor=:red)
                plot(p1, p2, p3, p4, p5, layout = (n, 1), size = (800, sz))
        
            end
        
        end

        display(plotPrices(prices, fundValue, BT, ET, numFund, numChart))
    end
end