#return  a single one hot vector of DNA

def AGCTtoArray(Nuc):
    onehot=[]
    if Nuc=="A" or Nuc=="a":
        onehot=(1, 0, 0, 0)
        return onehot
    elif Nuc=="G" or Nuc=="g":
        onehot=(0, 1, 0, 0)
        return onehot
    elif Nuc=="C" or Nuc=="c":
        onehot=(0, 0, 1, 0)
        return onehot
    elif Nuc=="T" or Nuc=="t":
        onehot=(0, 0, 0, 1)
        return onehot
    elif Nuc=="N" or Nuc=="n":
        onehot=(0, 0, 0, 0)
        return onehot
    else: 
        pass

#a function to convert AGCTN to 4d array
def AGCTtoArray2(Seq):
    onehot=[]
    for Nuc in Seq:
        if Nuc=="A" or Nuc=="a":
            onehot.append((1, 0, 0, 0))
            
        elif Nuc=="G" or Nuc=="g":
            onehot.append((0, 1, 0, 0))
        elif Nuc=="C" or Nuc=="c":
            onehot.append((0, 0, 1, 0))
        elif Nuc=="T" or Nuc=="t":
            onehot.append((0, 0, 0, 1))
        elif Nuc=="N" or Nuc=="n":
            onehot.append((0, 0, 0, 0))
        else: 
            pass
    
    return onehot