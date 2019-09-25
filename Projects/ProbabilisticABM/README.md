# Probabilistic ABM (Pre-Alpha)

This project explores the application of modern probabilistic programming languages on agent tracking and prediction.

Civil emergencies such as flooding, terrorist attacks, fire, etc., can have devastating impacts on people, infrastructure, and economies. Knowing how to best respond to an emergency can be extremely difficult because building a clear picture of the emerging situation is challenging with the limited data and modelling capabilities that are available. Agent-based modelling (ABM) is a field that excels in its ability to simulate human systems and has therefore become a popular tool for simulating disasters and for modelling strategies that are aimed at mitigating developing problems. However, the field suffers from a serious drawback: models are not able to incorporate up-to-date data (e.g. social media, mobile telephone use, public transport records, etc.). Instead they are initialised with historical data and therefore their forecasts diverge rapidly from reality.

To address this major shortcoming, this new research project will develop dynamic data assimilation methods for use in ABMs. These techniques have already revolutionised weather forecasts and could offer the same advantages for ABMs of social systems. There are serious methodological barriers that must be overcome, but this research has the potential to produce a step change in the ability of models to create accurate short-term forecasts of social systems.

## Project aims
 * To understand the utility of modern probabilistic programming languages (PPLs).
 * Apply PPLs to model a simple singular agent.
 * Perform data assimilation to calibrate and optimise inference on our agent’s behaviour.



## Installation

This project is being developed using Anaconda Distribution with Python 3.7. All versions for software packages can be found in [Requirements_Pyro.txt](https://github.com/Urban-Analytics/dust/blob/ProbablisticABM/Projects/ProbablisticABM/super_simple_model/requirements_Pyro.txt) Though this is subject to change during development. Currenly, the easiest way to install run these commands in turn.



Then you will need to create your environment;
```
conda update conda
conda create --name <your_environment_name>
conda activate <your_enviroment_name>
```
Then install the necessary software. Note that some of these are not part of conda's main channel and need to be installed from conda-forge or using pip.
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

