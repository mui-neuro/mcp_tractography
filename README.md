### Tractography of the Middle Cerebellar Peduncle using MRtrix3

This script performs tractography of the middle cerebellar peduncle (MCP) using alignment to fractional anisotropy (FA) population template and masking. It was implemented specifically for robust usage in diffusion data of clinical quality.

For complete usage, please see
```python
./mcp_tractography -h
```

Individual subject data (e.g., extracted using [dcm2niix](https://github.com/rordenlab/dcm2niix)) must be organized in the main directory as follow:
```
mrtrix/subject
    dwi_raw.nii.gz
    info.json
    bvecs
    bvals
```

The FA population template and masks to be extracted in the main directory can be downloaded from:
https://download.i-med.ac.at/neuro/archive/population_template.tar.gz

Alternative methods for performing tractography of the MCP:
https://github.com/MIC-DKFZ/TractSeg
https://dmri.mgh.harvard.edu/tract-atlas/

If you use this code, please reference the following paper:
Beliveau, V., Krismer, F., Skalla, E., Schocke, M. M., Gizewski, E. R., Wenning, G. K., Poewe, W., Seppi, K., & Scherfler, C. (2021). Characterization and diagnostic potential of diffusion tractography in multiple system atrophy. Parkinsonism & Related Disorders, 85, 30–36. https://doi.org/10.1016/j.parkreldis.2021.02.027
