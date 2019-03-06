import pyBigWig as pbw
import math
import glob as gl
def merge_biwig(bigwig_file_list, out_file):
    bigwig_list=[]
    _chromosome_list=[]
    _value_list=[]
    _header_list=[]
    start_list=[]
    end_list=[]
    for f in bigwig_file_list:
        _tmp_wig=pbw.open(f)
        chroms=_tmp_wig.chroms()
        print(chroms)
        for chrom_name, chrom_length in chroms.items():
            _header_list.append((chrom_name, chrom_length))
            j=0
            for s,e, v in _tmp_wig.intervals(chrom_name, 0, chrom_length):
                j+=1
                start_list.append(s)
                end_list.append(e)
                _value_list.append(v)
            
            _chromosome_list.append([chrom_name]*j)
            print(j)
        _tmp_wig.close()
    out_bigwig=pbw.open(out_file)
    out_bigwig.addHeader(_header_list)
    out_bigwig.addEntries(_chromosome_list, start_list, ends=end_list, values=_value_list)
    out_bigwig.close()

def main():
    file_list=gl.glob("/home/onimaru/fast2/1000genome/tmp/GRCh38_edited_ctcf_test_class_mm10_CTCF_intestine_0days_ENCFF464ZPC_rep2_chr*.bw")
    ofile="/home/onimaru/fast2/1000genome/tmp/tmp.bw"
    merge_biwig(file_list, ofile)
if __name__ == "__main__":    
    main()