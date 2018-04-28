
import os
print os.getcwd(), os.path.dirname(os.path.realpath(__file__))
with open("./test.txt", 'w') as fo:
    fo.write(os.getcwd()+" "+os.path.dirname(os.path.realpath(__file__)))