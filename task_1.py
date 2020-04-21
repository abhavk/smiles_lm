import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

sub_df = pd.read_csv('https://covid.postera.ai/covid/submissions.csv')
sub_df = sub_df[['SMILES', 'CID', 'creator', 'rationale', 'MW']]

def rowToclog(pandasrow):
    return Descriptors.MolLogP(Chem.MolFromSmiles(pandasrow['SMILES']))
    
def rowToHBD(pandasrow):
    return Descriptors.NumHDonors(Chem.MolFromSmiles(pandasrow['SMILES']))
    
def rowToHBA(pandasrow):
    return Descriptors.NumHAcceptors(Chem.MolFromSmiles(pandasrow['SMILES']))
    
def calculateViolations(pandasrow):
    violations = 0
    if pandasrow['HBD'] > 5:
        violations = violations+1
    if pandasrow['HBA'] > 10:
        violations = violations+1
    if pandasrow['MW'] >= 500:
        violations = violations+1
    if pandasrow['cLogP'] > 5:
        violations = violations+1
    return violations
    
def rowToSimilarity(comparing_fp, pandasrow, similarityfunc):
    mol = (Chem.MolFromSmiles(pandasrow['SMILES']))
    mol_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024))
    return similarityfunc(comparing_fp, mol_fp)
    
def onlyA(a, b):
    count = 0
    for i in range(len(a)):
        if (a[i] == 1):
            if (b[i] == 0):
                count=count+1
    return count

def onlyB(a, b):
    return onlyA(b,a)
    
def bothAB(a,b):
    count = 0
    for i in range(len(a)):
        if (a[i] == 1):
            if (b[i] == 1):
                count = count+1
    return count

def neitherAB(a,b):
    return len(a) - (onlyA(a,b) + onlyB(a,b) + bothAB(a,b))
                
def mod(a):
    count = 0
    for i in range(len(a)):
        if (a[i]):
            count = count+1
    return count
    
def simEuclidean(a_arr, b_arr):
    val = (bothAB(a_arr, b_arr) + neitherAB(a_arr,b_arr))/len(a_arr)
    return np.sqrt(val)
    
def simManhattan(a_arr, b_arr):
    val = (onlyA(a_arr,b_arr) + onlyB(a_arr,b_arr))/len(a_arr)
    return (1-val)
    
def simTanimoto(a_arr, b_arr):
    val = bothAB(a_arr, b_arr)/(mod(a_arr)+mod(b_arr)-bothAB(a_arr, b_arr))
    return val
    
def simCosine(a_arr, b_arr):
    val = bothAB(a_arr, b_arr)/np.sqrt(mod(a_arr)*mod(b_arr))
    return val
    
ROF_df = pd.DataFrame(sub_df)
ROF_df['cLogP'] = sub_df.apply(rowToclog, axis=1)
ROF_df['HBD'] = sub_df.apply(rowToHBD, axis=1)
ROF_df['HBA'] = sub_df.apply(rowToHBA, axis=1)
less_one_violations = (ROF_df.apply(calculateViolations, axis=1) <=1)*1
ROF_df['ROF_violations'] = less_one_violations

SARS_mol = Chem.MolFromSmiles('O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2ccccc2)CC1')
SARS_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(SARS_mol,3,nBits=1024))

euclideans = ROF_df.apply(lambda x: rowToSimilarity(SARS_fp, x, simEuclidean), axis=1)
manhattans = ROF_df.apply(lambda x: rowToSimilarity(SARS_fp, x, simManhattan), axis=1)
tanimotos = ROF_df.apply(lambda x: rowToSimilarity(SARS_fp, x, simTanimoto), axis=1)

def topSMILES(array,df):
    indices = array.argsort()[-5:][::-1]
    topSmiles = list()
    for index in indices:
        topSmiles.append((df.loc[index, 'SMILES'],array[index]))
    return topSmiles
    
print("Top 5 similar molecules based on euclidean distance = ")
print(topSMILES(euclideans, ROF_df))
print("Top 5 similar molecules based on manhattan distance = ")
print(topSMILES(manhattans, ROF_df))
print("Top 5 similar molecules based on tanimoto distance = ") 
print(topSMILES(tanimotos, ROF_df))