# Antidepressants Variability Ratio meta-analysis

This repo contains the code to reproduce the results of the paper
"Is there evidence of an individual treatment response of antidepressants in major depressive disorder?
A Bayesian meta-analysis", Volkmann-Volkmann-Mueller.

## Installation - using Docker (recommended)
In order to rerun this analysis the following two programs need
to be installed:
- [Docker](https://docs.docker.com/)
- `make`. For installation on MacOS or Linux we recommend to
first install [brew](https://brew.sh/) and then run `brew install make`.

### Running the analysis

Open a terminal window, go to the root folder of this repository,
and execute the following commands:

    $ make build  # this creates the Docker image for the analysis  

    $ make run  # this creates a Docker container that hosts an instance of Jupyter lab
    
Paste the following URL into a browser:

    http://127.0.0.1:8765/lab
    
Click on the notebook `main.ipdb` and run it.
    
## Conda users

Execute the following commands in your terminal

    $ conda create --name ad_meta python=3.7 -y
	$ source activate ad_meta
    $ python setup.py install
    
Now you can import all the functions used for the analysis.


## Citations
The data set in the folder `/data` is taken from
[Cipriani et al_GRISELDA_Lancet 2018_Open dataset](https://data.mendeley.com/datasets/83rthbp8ys/2)
under the Creative Commons Attribution 4.0 International Licence.