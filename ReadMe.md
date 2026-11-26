# Diffusion Visual Counterfactual Explanations for Semantic Segmentation Models

Code used for my Master thesis.
In this project I investigated the possibilities for creating visual counterfactuals for semantic segmentation models using diffusion model for realistic image modifications.

## How does it work
The general concept is based on the work of Augustin et al. [1]. They created the counterfactual by guiding the diffusion process in the direction of the wanted classification model output. 
In my work I replaced the classification model with a segmentation U-Net and added some methods for noice reduction and focusing the gradient on important part for the segmentation.
This why the pipeline was able to create visual counterfactuals that give an understandable explanations for segmentation error.

![Overview over the counterfactual genetation process](/assets/overview.jpg "Overview over the counterfactual genetation process")


## Train the models
All used segmentation models and diffusion models were trained on the CelebAMask dataset [2]. 

### Diffusion model

The diffusion model can be trained either with conditioning on the target model or without it. For the training the `experiments.train_diffusion` script can be used:

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=src python -m experiments.train_diffusion \
  data=celebamask \
  diffusion.epochs=<number of epochs> \
  diffusion.name=<Name of the diffusion model> \
  diffusion.use_target=<Use target mask for conditioning>
```

### Segmentation model
The segmentation model can be trained either with a normal segmentation training or additionally with the robust regularization training so it is robust against adversarial attacks. Additionally, on the CelebAMask dataset [2], the training skript allows for a training only on a subsection of the training set. For this the hyperparameter  For the training the `experiments.train_diffusion` script can be used:

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=src python -m experiments.train_segmentation \
  data=celebamask \
  segmentation.epochs=<number of epochs> \
  segmentation.name=<name of the segmentation model> \
  segmentation.robust.enable=<activate robust training> \
  segmentation.ood.enable=<activate subsection training> \
  segmentation.ood.attribute=<select attribute> \\
  segmentation.ood.train_on_positive=<select positive of negative group>
```


## Run the counterfactual generation

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=src python -m experiments.generate_counterfactual \
    diffusion.name=<Name of the diffusion model> \
    diffusion.use_target=False \
    data=celebamask \
    generation.model=<name of the target model> \
    generation.robust=<name of the robust model> \
    generation.sample_name=<sample name> \
    generation.name=<folder sample name> \
    generation.batch_name=<folder sample name> \
    generation.seed=<random seed> \
    generation.weight_segmentation=<weight for segmentation loss> \
    generation.weight_distance=<weight for similarity loss> \
    generation.AdaptGrad.N=<number of samples in AdaptGrad> \
    generation.approximation_step_size=<step size for approximation>  \
```


```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=src python -m experiments.generate_counterfactual \
    generation=baseline \
    data=celebamask \
    generation.model=<name of the target model> \
    generation.robust=<name of the robust model> \
    generation.sample_name=<sample name> \
    generation.name=<folder sample name> \
    generation.seed=<random seed> \
    generation.batch_name=<folder sample name> \
    generation.AdaptGrad.N=<number of samples in AdaptGrad> 
```


## References
[1] Augustin, Maximilian, et al. "Diffusion visual counterfactual explanations." Advances in Neural Information Processing Systems 35 (2022): 364-377.

[2] Lee, Cheng-Han, et al. "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.