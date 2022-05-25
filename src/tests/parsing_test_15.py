def czesc_wspolna(lst):
    pusty = None

    if lst == []:
        return pusty
    
    wyn = lst[0]
    for prost in lst[1:]:
        wyn = (  # taka forma, bo nie można przypisywać na elementy krotki
                max(wyn[0], prost[0]),
                max(wyn[1], prost[1]),
                min(wyn[2], prost[2]),
                min(wyn[3], prost[3]),
              )

    if wyn[0]>wyn[2] or wyn[1]>wyn[3]:
        return pusty
    else:
        return wyn
        
czesc_wspolna([(0,0,2,1),(3,2,5,3)])
