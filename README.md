# CoCoReco-ECCV2024
This is the code base for our **ECCV 2024 paper** "Connectivity-Inspired Network for Context-Aware Recognition" at the *"Human-inspired Computer Vision"* International Workshop, September 29, 2024, Milan.

### Abstract
The aim of this paper is threefold. We inform the AI practitioner about the human visual system with an extensive literature review; we propose a novel biologically motivated neural network for image classification; and, finally, we present a new plug-and-play module to model context awareness. We focus on the effect of incorporating circuit motifs found in biological brains to address visual recognition. Our convolutional architecture is inspired by the connectivity of human cortical and subcortical streams, and we implement bottom-up and top-down modulations that mimic the extensive afferent and efferent connections between visual and cognitive areas. Our Contextual Attention Block is simple and effective and can be integrated with any feed-forward neural network. It infers weights that multiply the feature maps according to their causal influence on the scene, modeling the co-occurrence of different objects in the image. We place our module at different bottlenecks to infuse a hierarchical context awareness into the model. We validated our \textbf{Co}nnectivity-Inspired \textbf{Co}ntext-Aware \textbf{Reco}gnition~(CoCoReco) network through image classification experiments on benchmark data and found a consistent improvement in performance and the robustness of the produced explanations via class activation.

## Code and Dataset
### Get started with the coding!
You can easily utilize our SLURM 'sbatch' submission file, [slurm_submit.x](https://github.com/gianlucarloni/CoCoReco-ECCV2024/blob/main/slurm_submit.x). That file sets some variables and launches the Python/Pytorch training script, [train.py](https://github.com/gianlucarloni/CoCoReco-ECCV2024/blob/main/train.py).

### CoCoReco
In [network.py](https://github.com/gianlucarloni/CoCoReco-ECCV2024/blob/main/model/network.py), you can find our novel **CoCoReco model**, depicted in this figure:
<img src="./readme_images/cocoreco.pdf" width=300>

In addition, [network.py](https://github.com/gianlucarloni/CoCoReco-ECCV2024/blob/main/model/network.py) includes our proposed the **Contextual Attention Block (CAB)**, which infers weights that multiply the feature maps according to their causal influence on the scene, modeling the co-occurrence of different objects in the image. 
<img src="./readme_images/CAB-module.pdf" width=300>


In case you find any issues related to the Dos2Unix conversion (when a file is created on Windows and used in Linux systems), you can easily convert it with this [online tool](https://toolslick.com/conversion/text/dos-to-unix).

### Dataset
In this work, we used *ImagenetteV2*, a smaller version of the popular Imagenet dataset, composed by the images corresponding to the 10 more easily classified classes.
You can find additional information on this dataset at [this page](https://github.com/fastai/imagenette?tab=readme-ov-file#imagenette-1). To download the "320 px" version, as we did, just download this [*.tgz*](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) file. If you are a Linux user, you can easily get that file from the command line interface with

```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
```

