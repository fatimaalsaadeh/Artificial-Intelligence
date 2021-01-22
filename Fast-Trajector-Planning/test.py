for item1 in range(len(basket)):
        for item2 in range(item1+1, len(basket)):
            # Note for a pair to have a support greater than 100, the individual items in the pair should have a support greater than 100. 
            # So filtering based on that rule.
            if (basket[item1] in itemswithsupport) and (basket[item2] in itemswithsupport):    
                # generating the pair based on the alphabetical ordering 
                if basket[item1]<basket[item2]:
                    pairofitem=(basket[item1], basket[item2])
                else:
                    pairofitem=(basket[item2], basket[item1])
                if pairofitem in paircount:
                    paircount[pairofitem]=paircount[pairofitem]+1
                else:
                    paircount[pairofitem]=1
