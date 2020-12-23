# Particle swarm optimization of deep neural networks architectures for image classification

**Authors:** Francisco Erivaldo Fernandes Junior and Gary G. Yen

This code can be used to replicate the results from the following paper:

F. E. Fernandes Junior and G. G. Yen, “**Particle swarm optimization of deep neural networks architectures for image classification**,” Swarm and Evolutionary Computation, vol. 49, pp. 62–74, Sep. 2019.

```
@article{fernandes_junior_particle_2019,
	title = {Particle swarm optimization of deep neural networks architectures for image classification},
	volume = {49},
	issn = {22106502},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S2210650218309246},
	doi = {10.1016/j.swevo.2019.05.010},
	language = {en},
	urldate = {2019-07-06},
	journal = {Swarm and Evolutionary Computation},
	author = {Fernandes Junior, Francisco Erivaldo and Yen, Gary G.},
	month = sep,
	year = {2019},
	pages = {62--74},
}
```

**Note1:** If your system has all these packages installed, the code presented here should be able to run on Windows, macOS, or Linux.

## Installation using Anaconda Python

Make sure you have Anaconda installed in your system. Then, run each command in a terminal window:

```
git clone https://github.com/feferna/psoCNN.git

conda env create -f psoCNN_env.yml

conda activate psoCNN
```

**Note1:** If your system has all the packages listed in the file psoCNN_env.yml, the code presented here should be able to run on Windows, macOS, or Linux.

**Note2:** The file psoCNN_env.yml installs the Nvidia CUDA Toolkit and cuDNN library necessary to run the project in an Nvidia GPU.

## Usage

1. Download the following datasets and extract them to their corresponding folders inside the ```datasets``` folder:
	1. Convex: 
[http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip)
	2. Rectangles: [http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip)
	3. Rectangles with Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip)
	4. MNIST with Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip)
	5. MNIST with Random Noise as Background: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip)
	6. MNIST with Rotated Digits: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip)
	7. MNIST with Rotated Digits and Background Images: [http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip)


2. Now, you can test the algorithm by running the ```main.py``` file:

	```
	python main.py
	```

**Note2:** The algorithm's parameters can modified in the file ```main.py```.

**Note3:** due to our limited resources, we cannot provide any support to the code in this repository.

