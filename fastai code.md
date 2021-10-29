# lesson1 pets

```python
from fastai.vision import *
from fastai.metrics import error_rate

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
help(untar_data)
```

```
Help on function untar_data in module fastai.datasets:

untar_data(url: str, fname: Union[pathlib.Path, str] = None, dest: Union[pathlib.Path, str] = None, data=True, force_download=False) -> pathlib.Path
    Download `url` to `fname` if it doesn't exist, and un-tgz to folder `dest`.
```

```python
path = untar_data(URLs.PETS); path
```

```
PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet')
```

```python
path.ls()

[PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images'),
 PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/annotations')]

path_anno = path/'annotations'
path_img = path/'images'
```

```
fnames = get_image_files(path_img)
fnames[:5]
```

```
[PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/saint_bernard_188.jpg'),
 PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/staffordshire_bull_terrier_114.jpg'),
 PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Persian_144.jpg'),
 PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Maine_Coon_268.jpg'),
 PosixPath('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/newfoundland_95.jpg')]
```

```python
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```



```python

```

```

```

