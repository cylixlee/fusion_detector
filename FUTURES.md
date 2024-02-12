# Year of Loong Future Plan

<sub>2024/2/12 • Architecture Improve • Hyperparameters Tuning</sub>

Happy the Chinese year of Loong! As my graduation design work, [fusion detector](https://github.com/cylixlee/fusion_detector) has conducted the first round of training-testing routine. In this round, we've reached the main goals below:

- **Generate** adversarial examples;
- **Train** our adversarial detector, using pretrained models as feature extractors;
- **Test** the detection rate on generated adversarial examples.

However, we've left some obvious imperfection in those parts. As the arrival of the new year, we'll head to architecture improvement and better detection performance.

## Architecture Improvement

As I mentioned before, there's some obvious imperfection and that's very clear in the source code. Let's start from the `tools` directory to inspect code clutter and come up with corresponding solutions.

### Tools

`tools` directory contains scripts about packing sources for deployment, generating adversarial datasets, etc. There's some redundant and unnecessary code due to deadline-oriented programming.
> About the project layout, please see [README](README.md).

#### Deprecation:

- `clean_cache.py`: Clean Python build cache (`__pycache__` directories). This script will be deprecated in favor of cache-exclusive packing script.
- `pack_deploy.py`: Packs the source **and** data files. This script will be deprecated, because simply uploading the pack with data is not efficient when using cloud GPU plaforms. Use the cloud GPU platform suggested cloud storage platforms instead.
- `generate_normalized_cifar_batch.py`: Loads CIFAR-10 dataset with mean and std from thirdparty dependency `pytorch_cifar10` and saves one batch of it to file. This file was created to test the accuracies of pretrained models, and can be deprecated since the result matches expectation.

#### Modification:

- `pack_source.py`: Packs **only** the source. This script shall add an extra `filter` argument to `tarfile.add()` to achieve cache-exclusive packing.
- `tools_common.py`: Extracted logic on retrieve the project root path. May use another logic independent from modifying `sys.path` to reference source code from tools code.
- `generate_adversarial_cifar_batch.py`: Basically plays a similar role like the generate-batch script mentioned before. It takes a batch from the encapsulated dataset and saves it to a file.
  1. The **logic** of this script can be changed *thoroughly*. Taking from existing adversarial dataset makes little sense. We shall use this script to generate one mini-batch and test attack rate on it, and generate the whole dataset if the result is promising.
  2. Existing **dependency** on encapsulated dataset shoule be removed. Considering create a module in `src` which offers a universal hyperparameter setting for consistency between this script and that whole-dataset generating one.
