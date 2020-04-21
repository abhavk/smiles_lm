run after conda activation

```
python task_1.py
```

For part 2, there are two modes - one for training and one for testing. 

1. To train the model, run
```
python train.py
```

2. To test with a SMILES prefix 
```
python quick_test.py <SMILES prefix>

# example
> python quick_test.py CCCCOC1

Input is (C)NC(=O)C1
load model from ./model.bin

extension (C)NC(=O)C1=CC=C has value -0.7370002921670675
extension (C)NC(=O)C1 has value -2.53005051612854
extension (C)NC(=O)C1=C(Cl) has value -3.0377864773618057
extension (C)NC(=O)C1=C(F) has value -3.1517090791021474
extension (C)NC(=O)C1=CC(Cl has value -3.735019765794277
```