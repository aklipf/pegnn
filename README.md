# Equivariant Message Passing Neural Network for Crystal Material Discovery

## Instalation

### Requirements

* tested on python 3.8
* pytorch
* torch-geometric
* tensorboard
* numpy
* ase [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
* pymatgen [PyMatGen](https://pymatgen.org/)
* pandas
* tqdm


### Datasets

Datasets must be downloaded from [https://github.com/txie-93/cdvae](https://github.com/txie-93/cdvae)

## Training

Start training:

```bash
python train_autoencoder.py -v --dataset ./data/mp_20 --device cuda
```

Get help

```bash
python train_autoencoder.py -h
```
