# Tomography-Processing-Pipeline
One-click processing from raw frames into denoised tomograms. 

# Workflow 
This script chains MotionCor3, imod newstack, AreTomo2 and cryoCARE for fully automated tomogram half-map reconstruction and denoising.

# Requirements
- https://bio3d.colorado.edu/imod/
- https://github.com/czimaginginstitute/MotionCor3
- https://github.com/czimaginginstitute/AreTomo2
- https://github.com/juglab/cryoCARE_pip

# Installation
Please install the above softwares. 


# Usage
Enter the directory containing your raw frames.Clone the repository:
```git clone https://github.com/DavidKart/Tomography-Processing-Pipeline.git```

Unpack everything into your directory:
```mv Tomography-Processing-Pipeline/* .```

Edit ```run.py``` accoring to your needs - initialize the necessary parameters accordingly at the top. Importantly, add the paths to the executables. The cryoCARE executables should be in PATH.

You will find the denoised tomograms in ```cryoCARE/runForAll/denoised.rec```.

additional notes:
- For large datasets, probably a few (3-5) representative tomograms are enough for training. Select a few .mdoc files accordingly in ```mdocryoCARETrain```
- you can run separately: 1. Creation of tomograms and tomogram half maps from raw data 2. cryoCARE train 3. cryoCARE predict.
- If the prediction fails due to memory issues: Edit other/cryoCare_blueprint/predict_config.json and increase the values for "n_tiles". 
