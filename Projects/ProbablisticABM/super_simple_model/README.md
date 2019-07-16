# Probabalistic Experiments

This project explores the application of modern probablistic programming languages on agent tracking and prediction.

## Installation

This project is being developed using Anaconda Distribution with Python 3.7. All versions for sofware packages can be found in [Requirements_Pyro.txt](https://github.com/Urban-Analytics/dust/blob/ProbablisticABM/Projects/ProbablisticABM/super_simple_model/requirements_Pyro.txt) Though this is subject to change during development. The easiest way to install do this run these commands in turn;

First create your environment;
```
conda update conda
conda create --name <your_environment_name>
conda activate <your_enviroment_name>
```
Then install the nessisary software. Note that some of these are not part of conda's main channel and need to be installed from conda-forge or using pip.
```
conda install pytorch-cpu torchvision-cpu -c pytorch
pip install pyro-ppl
conda install imageio matplotlib
conda install -c conda-forge imageio-ffmpeg
```

## Checklist for Packages
1.[PyTorch](https://pytorch.org/), An open source deep learning platform that provides a seamless path from research prototyping to production deployment.

2.[Pyro](https://pyro.ai/), a universal probabilistic programming language (PPL) written in Python and supported by PyTorch on the backend.<br>

3.[Matplotlib](https://matplotlib.org), a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

3.[Imageio](https://imageio.github.io/) a Python library that provides an easy interface to read and write a wide range of image data, including animated images, video, volumetric data, and scientific formats. **For the ability to render output as a video media format you will need to install the ffmpeg extension via the conda-forge channel.

## Project Intern

  **Benjamin Isaac Wilson** 
  [Github](https://github.com/BenjaminIsaac0111)<br>
  benjamintaya0111@gmail.com<br>
  medbwila@leeds.ac.uk

## Project Supervisors

  **Professor Nick Malleson** - (N.S.Malleson@leeds.ac.uk)<br>
  **Dr Jon Ward** - (J.A.Ward@leeds.ac.uk)
  
## Dust Project Sites
* **Data Assimilation for Agent-Based Models (DUST)** - (https://dust.leeds.ac.uk/)<br>
* **Data Assimilation for Agent-Based Modelling** - (https://urban-analytics.github.io/dust/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

