# CoCoReco-ECCV2024
This is the code base for our **ECCV 2024 paper** "Connectivity-Inspired Network for Context-Aware Recognition" at the *"Human-inspired Computer Vision"* International Workshop, September 29, 2024, Milan.

## Dataset
In this work, we used *ImagenetteV2*, a smaller version of the popular Imagenet dataset, composed by the images corresponding to the 10 more easily classified classes.
You can find additional information on this dataset at [this page](https://github.com/fastai/imagenette?tab=readme-ov-file#imagenette-1). To download the "320 px" version, as we did, just download this [*.tgz*](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) file. If you are a Linux user, you can easily get that file from the command line interface with

```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
```

## Code
To get started with the coding, you can easily utilize our SLURM 'sbatch' submission file, slurm_submit.x. In case you find any issues related to the Dos2Unix conversion (when a file is created on Windows and used in Linux systems), you can easily convert it with this [online tool](https://toolslick.com/conversion/text/dos-to-unix).
