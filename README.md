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

Edit ```run.py``` accoring to your needs - initialize the necessary parameters according at the top. Importantly, add the paths to the executables. The cryoCARE executables should be in PATH.

