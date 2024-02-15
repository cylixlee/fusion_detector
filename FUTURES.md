# Year of Loong Future Plan

<sub>2024/2/12 • Architecture Improve • Hyperparameters Tuning</sub>

- [Year of Loong Future Plan](#year-of-loong-future-plan)
  - [Architecture Improvement](#architecture-improvement)
    - [Tools](#tools)
      - [Deprecation](#deprecation)
      - [Modification](#modification)
      - [**Speial**: Adversarial Dataset Generating](#speial-adversarial-dataset-generating)
    - [Source](#source)
      - [Entrypoints](#entrypoints)
      - [Packages](#packages)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Adversarial Attack](#adversarial-attack)
    - [Feature Fusion](#feature-fusion)


Happy the Chinese year of Loong! As my graduation design work, [fusion detector](https://github.com/cylixlee/fusion_detector) has conducted the first round of training-testing routine. In this round, we've reached the main goals below:

- **Generate** adversarial examples;
- **Train** our adversarial detector, using pretrained models as feature extractors;
- **Test** the detection rate on generated adversarial examples.

However, we've left some obvious imperfection in those parts. As the new year arrives, we'll head to architecture improvement and better detection performance.

## Architecture Improvement

As I mentioned before, there's some obvious imperfection and that's very clear in the source code. Let's start from the `tools` directory to inspect code clutter and come up with corresponding solutions.

### Tools

`tools` directory contains scripts about packing sources for deployment, generating adversarial datasets, etc. There's some redundant and unnecessary code due to deadline-oriented programming.
> About the project layout, please see [README](README.md).

#### Deprecation

- `clean_cache.py`: Clean Python build cache (`__pycache__` directories). This script will be deprecated in favor of cache-exclusive packing script.
- `pack_deploy.py`: Packs the source **and** data files. This script will be deprecated, because simply uploading the pack with data is not efficient when using cloud GPU plaforms. Use the cloud GPU platform suggested cloud storage platforms instead.
- `generate_normalized_cifar_batch.py`: Loads CIFAR-10 dataset with mean and std from thirdparty dependency `pytorch_cifar10` and saves one batch of it to file. This file was created to test the accuracies of pretrained models, and can be deprecated since the result matches expectation.

#### Modification

- `pack_source.py`: Packs **only** the source. This script shall add an extra `filter` argument to `tarfile.add()` to achieve cache-exclusive packing.
- `tools_common.py`: Extracted logic on retrieve the project root path. May use another logic independent from modifying `sys.path` to reference source code from tools code.
- `generate_adversarial_cifar_batch.py`: Basically plays a similar role like the generate-batch script mentioned before. It takes a batch from the encapsulated dataset and saves it to a file.
  1. The **logic** of this script can be changed *thoroughly*. Taking from existing adversarial dataset makes little sense. We shall use this script to generate one mini-batch and test attack rate on it, and generate the whole dataset if the result is promising.
  2. Existing **dependency** on encapsulated dataset should be removed. Considering create a module in `src` which offers a universal hyperparameter setting for consistency between this script and that whole-dataset generating one.

#### **Special**: Adversarial Dataset Generating

This is the most special part, and needs thorough change.

- `generate_adversarial_dataset.py`: This script is used to generate adversarial dataset using some adversarial attacks. However it saves those generated Tensors directly, causing duplicate initialization of CUDA when running on GPU.
- `convert_adversarial_tensors.py`: Due to previous error, this script aimed to convert raw tensor files into PIL images and integers, compatible with the original CIFAR dataset. But when used to train detector along with the normal CIFAR dataset, vibration overfitting happens. `DataLoader(shuffle=True)` solves this but leaves a problem of frequently loading data segments, leading to poor performance.
- `generate_hybrid.py`: Loads previously generated adversarial **image** set and shuffle it with the normal one. Doing so solves the vibration overfitting and performance issues, but results in huge dataset size.

We can understand this routine by drawing a flowchart as below:

```
(Generate Adversarial Dataset)  =>  (Convert Tensors)  =>  (Shuffle)
               |                            |                  |
    Raw Adversarial Tensors       Adversarial Imageset   Hybrid Imageset
```

There's some apparent redundant and unnecessary work. The code contains a lot of *segment-file saving* and *tensor-image converting* operations, and that's ineffective and useless.

We only need to generate the adversarial imageset **once**:

1. Drop the *generate-convert* logic. Just use a universal hyperparameter setting mentioned previously to generate an adversarial imageset.
2. Deprecate the usage of hybrid imageset because that's too storage-demanding. Just creates a wrapper class extends `Dataset` and returns items from different dataset each time.

That shall solve the reinitialization and storage occupation issues.

### Source

Now comes the source part. Let's start from the entrypoints.

#### Entrypoints

- `test_baselines.py`: This script is for testing pretrained models' classification rate and can be deprecated since the accuracies match expectation.
- `train_detector.py`: The main logic of training the detector. The detector class is defined within this script using `pytorch-lightning`, and should be decoupled into other modules.

#### Packages

Speaking of packages, this project contains four:

1. `dataset`: contains the encapsulated dataset classes, the universal base class of which offers abstract method to get access to train set and test set separately. Due to previous adversarial example generating issues, this part should be overwritten from the ground up.
   - `NormalizedCifarDataset` log can be reserved. Other encapsulation classes can be deleted because of previous dataset generating issue.
   - When generated new adversarial dataset, create a new encapsulation correspondingly.
   - A helper class like `JaggedDataset` can be created in favor of train-time shuffling.
2. `extractor`: This may be the most relax part as the code logic is basically right and needs few changes. What we need is to add different kinds of extractors, extending the common abstract base class of course. The base class offers `extract()` method to, literally, extract features from existing pretrained models.
   - Note that the base class is not a derived class of `nn.Module` because extractor is often not trainable.
   - Remember to call the constructor of the base class, which sets the `requires_grad` attribute of underlying modules to `False` just to be safe.
3. `misc`: Helper classes useful throughout the whole project. My advice is just let them being, but here's indeed some improvements can be made by:
   - `hook.py`: The logic of getting a certain layer of an existing model. A context-manager class called `LayerOutputValueColletor` is created, which adds a hook function to the specified layer when entering the context and removes it when exits. Maybe the frequently adding and removing hook operation will cause performance degradation, so other solutions like taking advantage of `pytorch-lightning` can be adopted somehow.
   - `utility.py`: The entire module can be deleted, since most of its content is not referenced. The `PossibleRedirectStream` logic can be inlined into the summary and structure inspection functions.
   - Other modules can be removed directly.
4. `thirdparty`: Thirdparty source libs from open-source platforms (mostly GitHub). For now there's only `pytorch-cifar10` since it is very useful but not so common to upload to PyPI. Those thirdparty libraries often need no modification and the only necessary part, **path redirecting**, is completed, so no changes will be made here.

There's still two core modules of this project without a directory.

1. `classifier.py`: contains the binary classifier model. This module is not so important in improving the detection rate, since feature transforms are performed in `extractor` modules.
2. `perturbation.py`: the wrapper module of underlying adversarial attacks. The only place to rewrite might be the default parameters of the functions, and we just do hyperparameter settings in an individual module.

## Hyperparameter Tuning

> Despite the hard work of architecture improvement, this part may be the most uncontrollable and *painful* stage.

I'm not an expert in this so I'll just follow my supervisor's advice.

### Adversarial Attack

Currently the adversarial attacks are applied using some rough hyperparameters (epsilon, epsilon per iteration, number of iterations, etc.). Suggestions are below:

1. **epsilon**: epsilon should be set in the range of $[\frac{2}{255}, \frac{8}{255}]$, and can **not** exceed the limit of $\frac{16}{255}$.
2. **iterations**: number of iterations can be increased, since epsilon is decreased. We'll use a more human-unrecognizable but more powerful perturbation.
3. **epoch**: epoch count can be increased too.

> **Necessity**
>
> The necessity of improve the attack performance is that, if we use so-called "adersarial examples" that can only cause 20% to 30% error rate of the pretrained model, those examples are actually **still normal examples**.
>
> Adversarial examples must distinguish from the normal ones, or they will eventually affect detection rate: we programmers see them as adversarial and the model sees them as normal, thus $\displaystyle AttackSuccessRate=\frac{Misclassification\downarrow}{Adversarial\uparrow}$ because we treat some "normal examples with little noise" as adversarial ones.

### Feature Fusion

Convolution can be introduced in feature fusion, in order to improve detection performance.
