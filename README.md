# GME_tutorials
![Overview](tutorials/alanine_dipeptide/png/pipeline.jpg)  

Tutorials for non-Markovian dynamic models (qMSM and IGME) to study protein dynamics.

### Content
There are step-by-step Jupyter notebooks that build GME-based non-Markovian dynamic models (qMSM and IGME) in two peptide systems: the alanine dipeptide and villin headpiece.

For the alanine dipeptide system:

1. Featurization. [link to notebook](tutorials/alanine_dipeptide/Featurization.ipynb)
3. Cross-validation with GMRQ. [link to notebook](tutorials/alanine_dipeptide/Cross_Validation.ipynb)
4. Construction of Microstate-MSM. [link to notebook](tutorials/alanine_dipeptide/MicroMSM.ipynb) In this Jupyter notebook, Dimensionality Reduction, Geometric Clustering, and Microstate-MSM are included.
5. Construction of Macrostate-GME Model. [link to notebook](tutorials/alanine_dipeptide/MacroGME.ipynb) In this Jupyter notebook, Kinetic Lumping, Macrostate-qMSM, Macrostate-IGME, and Macrostate-MSM are included.

For the villin headpiece system, we apply two independent methods for Dimensioanality Reduction: tICA and SRVs. Either one is fine for this system. The Jupyter notebooks are as follows:

1. Featurization. [link to notebook](tutorials/villin_headpiece/Featurization.ipynb)
2. Feature Selection (by Spectral OASIS). [link to notebook](tutorials/villin_headpiece/Feature_Selection.ipynb)
3. Cross-validation with GMRQ.
   
   tICA version: [link to notebook](tutorials/villin_headpiece/tICA_Cross_Validation.ipynb)
   
   SRVs version: [link to notebook](tutorials/villin_headpiece/SRVs_Cross_Validation.ipynb)
4. Construction of Microstate-MSM.
   
   tICA version: [link to notebook](tutorials/villin_headpiece/tICA_MicroMSM.ipynb)
   
   SRVs version: [link to notebook](tutorials/villin_headpiece/SRVs_MicroMSM.ipynb) 
5. Construction of Macrostate-GME Model.
   
   Analysis after tICA: [link to notebook](tutorials/villin_headpiece/tICA_MicroMSM.ipynb)
   
   Analysis after SRVs: [link to notebook](tutorials/villin_headpiece/SRVs_MicroMSM.ipynb)


### Installation
We use [MSMbuilder2022](https://github.com/msmbuilder/msmbuilder2022) and [PyEMMA](https://github.com/markovmodel/PyEMMA/) 2.5.12 in our tutorial. 
For MSMbuilder2022, Anaconda, you can use the following script:

	conda install -c conda-forge testmsm

For installing MSMbuilder with pip, you can use the following command
For PyEMMA installation, you may refer to http://pyemma.org.

### Contact Information
You may contact Yue Wu wu678@wisc.edu/ Siqin Cao scao66@wisc.edu in case you have any specific questions on the model construction or the original MD data for the alanine dipeptide.
You may contact D. E. Shaw Research for the MD data of the villine headpiece.
