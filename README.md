# Fusion Detector

<sub>2024/2/12 • Project Introduction</sub>

This is my graduation design work, it's **Adversarial Examples Detector based on Multi-Model Feature Fusion** in essence.

## Project layout
Its more code-demanding than I initially thought, thus a decent project layout is required. The project is divided into several directories, each of which contains files differ from others':

- `data`: Pretrained models on a certain dataset and the dataset itself.
- `metadata`: Model structure and summary on parameters and output shapes.
- `src`: Source code of core logic and fundamentals.
- `tools`: Tools that help packing sources for deployment, generating adversarial datasets, etc.

> You may notice that some of the directories (e.g. the `data` directory) is missing from GitHub. Despite GitHub's file size limit, data files **shall not** be uploaded to git repositories.
>
> Since the project is still under development, you can email me ([cylix.lee@outlook.com](mailto:cylix.lee@outlook.com)) to get data files. Once the project is complete, those data files will be uploaded on a cloud storage platform.
