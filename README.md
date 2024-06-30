# CS1470 Final Project Report

## Introduction
The focus of our project is to reimplement stable diffusion. We will be reimplementing a Variational Autoencoder, UNet, and pretrained CLIP within our stable diffusion model in TensorFlow. We aim to generate high-resolution images conditioned on a text prompt using language model embeddings. This has many applications in high-quality content creation, such as ads, posters, and illustrations. Our challenge is to optimize this architecture for the limited computational resources we have at our disposal. Our work is inspired by the paper "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding," which presents Imagen, a text-to-image diffusion model that generates photos with an unprecedented degree of photorealism. We chose this paper to challenge ourselves to see how much of what they describe can be implemented with limited hardware, and we were interested in seeing what dataset we could successfully train such an architecture on, given that diffusion is typically considered a data-demanding architecture. We chose to reimplement stable diffusion instead of a simple diffusion model because we find it more interesting to apply the diffusion process over a lower-dimensional latent space, which is what stable diffusion does.



## Methodology


Our architecture comprises a Variational Autoencoder, a UNet, and a pretrained CLIP. We are training the model with the TensorFlow framework. The hardest part about implementing the model would be using the computational resources we have at our disposal to generate high-quality images. The paper generates 1024x1024 images, but we can aim for smaller images. We can use a smaller text encoder and try to simplify the architecture as much as possible while preserving quality outputs. Our reach idea is to implement text-to-3D model generation that can then be 3D printed. However, we decided to start with text-to-2D diffusion. Backup ideas include training the model on a smaller dataset after pre-training to generate outputs of a specific type specifically for content creators. We would achieve this by creating a custom labeled dataset using a neural network (https://github.com/salesforce/BLIP) to generate the captions for the images and train our model on this dataset. Alternatively, we would try other model architectures for the diffusion model and perhaps opt for simpler architectures.

## Results

We trained and tested our model on the Fashion MNIST dataset. Our model achieved an average loss across batches of 36.631 (on 15 batches), while the PyTorch paper we reimplemented achieved an average loss across batches of 41.481, so we managed to improve the given model. While blurry, the generated images still depict identifiable clothing items on the Fashion MNIST dataset.

<img width="844" alt="Screenshot 2024-06-30 at 12 13 57 AM" src="https://github.com/julia-fu0528/3Dfusion/assets/110797555/102dbf34-7281-4e35-8545-6c0967581b29">

Future improvements could include tuning the hyperparameters of the model or adding more layers. Our end goal was to create a 3D printable model, which is challenging given the computational power we had and the specific constraints of the architecture of a 3D printable model.

## Challenges

The biggest challenge we faced was tuning the hyperparameters. Because TensorFlow and PyTorch implemented their Conv2D and ConvTranspose2D differently, using the same stride size as in the PyTorch implementation does not work for TensorFlow in direct translation—shape errors occur because the padding is different. We are currently trying to figure out the correct hyperparameters to effectively train the model.

The current limitation is that the diffusion model is computationally expensive and therefore cannot run efficiently on our hardware. We need to parallelize it on a GPU, which is also challenging and hard to debug. Additionally, due to memory limitations, we cannot train the model using a large batch size as the source implementation did, which may contribute to the lower quality in the predictions.

## Reflection

We reached our base and target goals successfully. Our model outperforms the PyTorch paper we started from, so we feel that this project was successful. We did not end up implementing text-to-3D diffusion, which was the stretch goal, but we still achieved good results in our other goals. We could adapt our architecture to achieve good results on different datasets, try different data augmentations and preprocessing techniques, and possibly implement text-to-3D stable diffusion. We could also experiment with other transformer architectures and combine results from other papers to discover new findings. We learned how to implement diffusion and adapt a model architecture to work for different datasets. Given that stable diffusion is a very promising model architecture, we believe these skills will serve us well in future projects. Additionally, we learned about the challenges and limitations of using different datasets, as a significant part of our project involved deciding and testing our model on different datasets.
