# Overview

This repository hosts the SoyCult dataset along with the associated source codes, corresponding to the paper 'Towards Automatic Soybean Cultivar Identification: SoyCult Dataset and Deep Learning Baselines' authored by Eliezer Soares Flores, Marcelo Resende Thielo, and Fabio Ronei Rodrigues Padilha, which was published in the proceedings of the 36th Conference on Graphics, Patterns, and Images (SIBGRAPI 2023).

## Organization

- **data**: This folder contains all the datasets and data files used in the project. In particular, the images utilized as input for this work, originating from Fabio Ronei Rodrigues Padilha's master's thesis as described in **Section II.A** of the paper, can be located in the **data/imgs** subfolder. **<span style="color:red">The images corresponding to the SoyCult dataset can be found in the data/seeds subfolder.</span>**

- **seg**: This folder contains the Python scripts for the contour-based segmentation of the soybean seeds, as described in **Subsection II.B** of the paper. To reproduce the results presented in the paper, execute these scripts in the following order:

  1) **imgs2edges.py**:
     - Inputs: Images in 'data/imgs'
     - Outputs: Images in 'data/edges' and 'data/edges_overlayed'

  2) **edges2conts.py**:
     - Inputs: Images in 'data/edges' and 'data/imgs'
     - Outputs: Images in 'data/conts' and 'data/edges_overlayed'

  3) **conts2holes.py**:
     - Inputs: Images in 'data/conts'
     - Outputs: Images in 'data/holes'

  4) **holes2seeds.py**:
     - Inputs: Images in 'data/holes' and 'data/imgs'
     - Outputs: Images in 'data/seeds'

  Note: The script 'compute_maxshape.py' is solely used to determine the dimensions of the largest grain among all cultivars. Its result is already integrated as parameters in 'holes2seeds.py' and does not need to be executed.

- **pred**: This folder contains the Python scripts for soybean cultivar predictions using baseline systems, described in **Section III** of the paper, as well as the scripts for evaluating these systems, described in **Section IV** of the paper. To reproduce the results presented in the paper, execute these scripts in the following order:

  1) **gen_folds.py**:
     - Inputs: Images in 'data/seeds'
     - Outputs: CSV files in 'data/folds'
     
  2) **gen_seed_folds.py**:
     - Inputs: CSV files in 'data/folds' and images in 'data/seeds'
     - Outputs: Images in 'data/seeds_folds'
     
  3) **feature_extraction.py**:
     - Inputs: Images in 'data/seeds_folds'
     - Outputs: MAT-files in 'data/feats'
          
  4) **tests.py**:
     - Inputs: MAT-files in 'data/feats/densenet201'
     - Outputs: Plots in 'data/results/conf_matrices'  e   'data/results/post_hoc'
     
  Note: The script 'training.py' is solely used to tune the optimal hyperparameters for the top-level classifiers. As their results are already integrated in 'tests.py', it does not need to be executed. If executed, their results will be saved as Latex tables in 'data/tuning'.
     
## Dependencies

The scripts were developed using an Anaconda environment set on a Linux machine (with specifications provided in the last paragraph of **Section IV** of the paper). Machine/deep learning models were implemented using Scikit-Learn and Tensorflow/Keras. To recreate the project environment and properly execute the scripts, follow these steps:

1. **Clone the Repository:**
   git clone https://github.com/eliezersflores/SoyCult.git
   cd SoyCult
   
2. **Create and Activate the Environment:**
   conda env create -f environment.yml
   conda activate soycult  
   
3. **Navigate to Project Folders and Execute Scripts:**
You can now navigate to specific project folders, as outlined in the organization section of this README, and execute the scripts to reproduce the results.

4. **Deactivate the Environment (After Completion):**
When you're done, deactivate the environment using the following command.

## Questions?

If you come across any bugs, have questions, or would like to share your feedback, please don't hesitate to get in touch with me [eliezersflores@gmail.com].
