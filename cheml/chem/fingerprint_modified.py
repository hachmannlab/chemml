from sys import argv
from rdkit import DataStructs
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Fingerprints.FingerprintMols import FoldFingerprintToTargetDensity

import time
start_time = time.time()

args=argv

data=open(args[1], 'r')
data2=data.read()
data.close()
del data
smiles=data2.split('\n')
del data2
smiles.pop(-1)

#output=open('/gpfs/projects/hachmann/Mikhail/132117/output_dense.fpt', 'w')

stage_1_time=time.time()-start_time
print "My stage 1 took", time.time() - start_time, "to run"

prints=[RDKFingerprint(MolFromSmiles(x)) for x in smiles]

del smiles

stage_2_time=time.time()-stage_1_time-start_time
print "My stage 2 took", stage_2_time, "to run"

dense=[DataStructs.cDataStructs.FoldFingerprint(x,int(str(args[3]))) for x in prints]
#dense=[FoldFingerprintToTargetDensity(x, 0.95) for x in prints]

stage_3_time=time.time()-stage_1_time-stage_2_time-start_time
print "My stage 3 took", stage_3_time, "to run"

dense_similarity=[]

for i in range(1,len(dense)):
    result=DataStructs.FingerprintSimilarity(dense[0],dense[i])
    putting=(str(i),str(result))
    dense_similarity.append(putting)
    #output.write(str(putting))
    #output.write('\n')

stage_4_time=time.time()-stage_3_time-stage_2_time-stage_1_time-start_time
print "My stage 4 took", time.time() - start_time, "to run"
            

#output.close()

output2=open(args[2], 'w')

for i in dense_similarity:
    if float(i[1])>float(args[4]):
        ind=int(i[0])
        x=DataStructs.FingerprintSimilarity(prints[0],prints[ind])
        putting=i[0]+' '+str(x)+'\n'
        output2.write(putting)
    else:
        continue

if dense_similarity==[]:
    for i in dense_similarity:
        if float(i[1])>(float(args[4])-0.05):
            ind=int(i[0])
            x=DataStructs.FingerprintSimilarity(prints[0],prints[ind])
            putting=i[0]+' '+str(x)+'\n'
            output2.write(putting)
        else:
            continue

output2.close()

print "My stage 5 took", time.time()-stage_4_time-stage_3_time-stage_2_time-stage_1_time-start_time, "to run"
print "My program took", time.time() - start_time, "to run"




