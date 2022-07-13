def RemBalance(balance, annualInterestRate, monthlyPaymentRate):
    for i in range(12):
        new_var
    MonIntRt = annualInterestRate / 12
    Minmonpay = monthlyPaymentRate * balance
    MoUnbal = balance - (Minmonpay*i)
    newbal = MoUnbal + (MonIntRt * MoUnbal)
    return RemBalance(newbal, annualInterestRate, monthlyPaymentRate)
    print(newbal)

RemBalance(42, 0.2, 0.04)
