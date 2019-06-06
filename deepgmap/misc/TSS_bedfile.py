
gene_list=[]
with open ('/media/koh/HD-PCFU3/mouse/mouse_UCSC_wholegenes.bed', 'r') as fin:
    with open ('/media/koh/HD-PCFU3/mouse/mouse_TSS.bed', 'w') as fout:
        for line in fin:
            if not line=='' or not line=='\n':
                line=line.split()
                chromosome=line[0]
                left=int(line[1])
                right=int(line[2])
                direction=line[5]
                if direction=='+':
                    gene=line[0]+':'+line[1]
                elif direction=='-':
                    gene=line[0]+':'+line[2]
                    
                if not gene in gene_list:
                    gene_list.append(gene)
                    if direction=='+':
                        start=left-1000
                        end=left+1000
                        fout.write(str(chromosome)+'\t'+str(start)+'\t'+str(end)+'\n')
                    if direction=='-':
                        start=right-1000
                        end=right+1000
                        fout.write(str(chromosome)+'\t'+str(start)+'\t'+str(end)+'\n')
                