def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def div_roundup(x, y):
    if y%x==0:
        return y/x
    else:
        return y/x+1

