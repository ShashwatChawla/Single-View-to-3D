# **Steps to run code**

## Fit Model

```
python -m main --mode "fit" --type 'vox|mesh|point'
```

or

```
python fit_data.py --type 'mesh|vox|mesh'
```

> *Gifs will be saved at: 'results/'*

## Train Model

```
python -m main --mode "train" --type 'vox|mesh|point'
```

or

```
python train_model.py --type 'vox' 
```

## Evaluate Model

```
python -m main --mode "eval" --type 'vox|mesh|point' --load_checkpoint
```

or

```
python eval_model.py --type point|mesh|point --load_checkpoint
```

> *Gifs will be saved at: 'results/arg.type/'*

To look at another interpretation(2.6)

```
python eval_model.py --type point  --interpret --load_checkpoint
```

> *Gifs will be saved at: 'results/interpret/'*
