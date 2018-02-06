
position_set=set()

with open("/home/fast/onimaru/data/mm10_1000_limb_altwindow_75co_non.labeled", 'r') as f1, open("/home/fast/onimaru/data/mm10_1000_altwindow_non.fa", 'r') as f2:
    for line in f1:
        line=line.split()
        if int(line[3])==1:
            position_set.add(line[0]+":"+line[1]+"-"+line[2])
    
    with open("/home/fast/onimaru/data/mm10_1000_limb_altwindow_75co_non.labeled.fa", 'w') as fo:
        WRITE=False
        for line in f2:
            if line.startswith('>'):
                current_position=line.strip('>\n')
                if current_position in position_set:
                    fo.write(line)
                    WRITE=True
                else:
                    WRITE=False
            elif WRITE:
                fo.write(line)
                