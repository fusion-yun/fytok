# FyTok

## Description

FyTok is an integrated modelling simulator based on Tokamak Ontology (IMAS DD). The idea of ontology modelling is introduced in the design of FyTok, which makes it possible to manage and organise the dependencies of physics modules and build complex workflows based on the physics semantics. FyTok supports a flexible plug-in mechanism to interact with third-party physics programs.

## NOTICE：

- IMAS data dictionary(IMAS DD) is not included in this project, please download it from the official website(Requires authorisation from ITER).
- The base functionality and architecture in FyTok that is not related to specific physical semantics is provided by the project [SpDB (SpDM)](https://github.com/fusion-yun/spdm), which is a data integration tool that unifies different data sources in a standard data format for use in FyTok.



## Software Architecture

Software architecture description

![Image](docs/figures/fuyun.svg "FuYun")

## Functions and Features
FyTok uses the IMAS DD as its core data model, following its hierarchical tree structure. FyTok achieves the conversion from a static Ontology description to a dynamic computational model by binding specific numerical computation processes to the static tree structure to form a dynamic computational model.

FyTok employs a top-down approach, adopting a holistic view of the Tokamak。
- In IMAS DD, physical concepts are divided into different IDS (Interface Data Structure) based on their relevance. Closely related physical quantities are organised together to form individual IDSs.
- The evolution of each IDS is modelled to form a computational module.
    - In FyTok, IDSs are modelled objects, not just data interfaces in the traditional sense.
    - IDSs are classified into two categories, subsystems and physical concepts, depending on their characteristics.
        - Physical Concepts, such as "equilibrium", "transport solver" and "core source term",
        et al. The updating of the state of the physics concepts must be implemented by binding the corresponding simulation programs.
        - Subsystems, such as "current coils", "device walls" and "magnetic probes",et al. The subsystems describe the geometrical information of the device or the experimental diagnostic data, and their state is updated as a result of data mapping.
-  "Tokamak class" abstracts the entire tokamak modelling.
    - FyTok defines the "Tokamak" class as the primary entry point for the centralisation of coordination and management of submodules and subsystems. 
    - Dependencies exist between the physical concepts described by different ‘modules’, where the output of one ‘module’ is the input of another.
    - "Tokamak class" Combined workflows based on the dependencies described in the ontology to achieve integrated computation and complete the overall modelling of the tokamak.
- Third-party physics programs are integrated in a modular form.
## Installation guide

### Requirements and preparing the Installation Environment:
Install Python 3.11 or greater.

#### For Linux:
```
sudo apt install python3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100
sudo update-alternatives --config python3
```
####  For Windows 11：

- Windows Subsystem for Linux（WSL2） is recommended.
- install WSL,open PowerShell 
```{powershell}
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux 
wsl --set-default-version 2           
wsl --install -d Ubuntu-22.04        
wsl --list --online                  
wsl -l -v                            
wsl --set-version Ubuntu-22.04 2     
```
- install python in ubuntu
```
sudo apt-get install python3.11  python3-pip
```
#### Integrated Development Environment 
 - vscode is  recommended
 - Install Visual Studio Code on Windows (not in the WSL file system).
 - installs extension packages in VSCODE:
    - WSL 
    - Python
    - Jupyter
- Open VSCode and select Link to WSL in the bottom left corner.

### Install FyTok 
```{bash}
pip install fytok
```
- check install:
```{bash}
python -c "import fytok"
```
- test install:
    - ctrl+shif+p to open notebook
    - use the following command:

```{bash}
from fytok.tokamak import Tokamak
```

## Running examples
- refer to [fytok_tutorial](https://github.com/fusion-yun/fytok_tutorial)
### Normalized Data Organization and Access
- load basic enviroment 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from spdm.core.entry import open_entry
from spdm.core.function import Function
from spdm.view import sp_view
```
#### Universal Resource Locator URI for data access

```
<local schema>+<protocol>+<format>://<address>/<path>? <query>#<fragment>
```

Extending protocol

- schema : Specifies the data semantics, usually the device name.
- protocol : Specifies the data protocol, such as file or mdsplus.
- format : Specifies the file format explicitly

standard semantics

- address : Specifies the data address, e.g., IP or hostname.
- path : Specifies the data path
- query : Specifies a data query condition
- fragment : Specifies a fragment of data

#### example for device data access
-  Accessing Device Data,such as wall for EAST Tokamak
```
from fytok.modules.wall import Wall

wall = Wall("east://#wall")
```
- get the data in wall 
```
# The access path strictly follows the IMAS DD.
desc = wall.description_2d[0]
r = wall.description_2d[0].limiter.unit[0].outline.r
```
- view the wall data 
```
fig = sp_view.display(wall)
```
- Similarly, access to other device data is supported, such as ITER.
```
fig = sp_view.display(Wall("iter://#wall"))
```
#### example for file data  access
- for geqdsk file 
- geqdsk.txt is a result file calculated by other equilibrium calculation programs that holds data for a particular equilibrium state. The file saves the data in a private geqdsk format.
```
eqdsk_file = open_entry(f"file+geqdsk://./data/geqdsk.txt")
psi_norm = eqdsk_file.get("equilibrium/profiles_1d/psi_norm")
## get f
f_value = eqdsk_file.get("equilibrium/profiles_1d/f")
```
### Third-party module integration
- There are two ways to integrate a new module, custom module and as a plugin.
- load basic enviroment 
```
import sys

# Add the path to the fytok_tutorial package to the python path
sys.path.append("${workdir}/fytok_tutorial/python")

import typing
import numpy as np
from spdm.view import sp_view

from fytok.utils.logger import logger
from fytok.modules.equilibrium import Equilibrium
from fytok.modules.core_transport import CoreTransport   
from fytok.modules.core_profiles import CoreProfiles 
```
#### Custom Modules 
- Demo for CoreTransport
    - write a Class for custorm demo ,which interite from basic class CoreTransport.Model in FyTok
    ```
    class CoreTransportDemo(CoreTransport.Model, code={"name": "demo"}):
        """Plugin Demo: CoreTransportModel"""

        def execute(
            self, *args, equilibrium: Equilibrium, core_profiles: CoreProfiles, **kwargs
        ) -> typing.Self:
            res = super().execute(
                *args,
                equilibrium=equilibrium,
                core_profiles=core_profiles,
                **kwargs,
            )
            res_1d: CoreProfiles.Profiles1D = res.profiles_1d
            eq_1d: Equilibrium.Profiles1D = equilibrium.profiles_1d
            prof_1d: CoreProfiles.Profiles1D = core_profiles.profiles_1d
            psi_norm = res_1d.grid.psi_norm
            rho_tor_norm = res_1d.grid.rho_tor_norm

            res.profiles_1d.conductivity_parallel = np.sin(rho_tor_norm * 6.28)
            return res


    spitzer = CoreTransportDemo()
    ```
    - init  CoreTransportDemo
    ```
    spitzer = CoreTransportDemo()
    ```
    - excute it 
    ```
    spitzer.refresh(equilibrium=eq, core_profiles=core_profiles)
    ```
  - view the result
  ```
  fig = sp_view.plot(
    spitzer.profiles_1d.rho_tor_norm, spitzer.profiles_1d.conductivity_parallel)
  ```
#### Plugin Modules 
- Plugin management 
    - Third-party physical applications do not need to be packaged within FyTok's framework. The user only needs to expose the directory of the encapsulated code to a path that can be retrieved by FyTok.
    - The directories are organised according to the following standard structure:
    ```
    {work_dir}/python/fytok/plugins/modules/< module type>/< physical module name>
    ```
    - {work_dir} is an arbitrary directory locally specified by the user
    - \<module type\> strictly adheres to the classification of physical concepts or device component descriptions in IMAS DD , commonly used are:
        - equilibrium
        - transport_slover
        - core_transport/model
        - core_sources/source
    - \<physical module name\>: name of the third-party physical programme,such as efit,freegs..
- Plugin call 
    - Initialise the plugin and specify the plugin name,such as spitzer_demo
    ```
    model = CoreTransport.Model(code={"name": "spitzer_demo"})
    ```
    - excute it 
    ```
    model.refresh(equilibrium=eq, core_profiles=core_profiles)
    ```
  - view the result
  ```
  fig = sp_view.plot(
    spitzer.profiles_1d.rho_tor_norm, spitzer.profiles_1d.conductivity_parallel)
  ```

###  Tokamak modelling
- Equilibrium analysis
```
from fytok.modules.equilibrium import Equilibrium
equilibrium = Equilibrium("file+geqdsk://./data/geqdsk.txt#equilibrium")
fig = sp_view.display(equilibrium)
```
- Create Tokamak
```
from spdm.view import sp_view
from fytok.contexts.tokamak import Tokamak
tokamak = Tokamak(
    device="iter",
    shot=900003,
    equilibrium="file+geqdsk://./data/geqdsk.txt#equilibrium",
)
```
- View Tokamak
```
print(tokamak)
fig = sp_view.display(tokamak)
```

## Contribution
- If there are any questions, you can contact us(yuzhi@ipp.ac.cn and lxj@ipp.ac.cn). 
- It is recommended to submit your questions directly within the issue system.
