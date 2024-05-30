import pathlib

import numpy as np
from spdm.core.file import File
from spdm.core.entry import Entry
from spdm.utils.logger import logger


def sp_read_iterdb_txt(path: str | pathlib.Path):
    """
    :param file: input file / file path
    :return: profile object
    """
    data = []
    with open(path, mode="r") as fid:
        fid.readline()
        key = None
        value = ""
        for line in fid.readlines():
            if not line.startswith("*"):
                value += line
                continue
            elif key is not None:
                d = []
                for s in value.split():
                    if s.isdigit():
                        d.append(int(s))
                    else:
                        try:
                            t = float(s)
                        except ValueError:
                            d.append(s)
                        else:
                            d.append(t)

                data.append([key] + d)

            key = line[1:].strip()
            value = ""

        # data.append((key,  [float(s) for s in value.split()]))
    return data


def sp_to_imas(data: dict):
    entry = Entry({})
    n = 0
    # fmt:off
    entry["shot"]                                                                                  = data[0][1]
    nx                                                                                        = data[n:=n+1][1]
    nion                                                                                      = data[n:=n+1][1]
    nprim                                                                                     = data[n:=n+1][1]
    nimp                                                                                      = data[n:=n+1][1]
    nneu                                                                                      = data[n:=n+1][1]
    nion_beam                                                                                 = data[n:=n+1][1]
    namep                                                                                     = data[n:=n+1][1:]
    namei                                                                                     = data[n:=n+1][1:]
    namen                                                                                     = data[n:=n+1][1:]
    entry["time"]                                                                                  = data[n:=n+1][1]
    entry["geometric_center/r"]                                                                    = data[n:=n+1][1]   # Rgeom : major radius of geometric center at elevation of magnetic axis, meters
    entry["magnetic_axis/r"]                                                                       = data[n:=n+1][1]   # major radius of magnetic axis, meters
    r0                                                                                        = data[n:=n+1][1]   # major radius of vacuum btor ref location, meters
    entry["equilibrium/vacuum_toroidal_field/r0"]                                                  = r0 
    entry["equilibrium/time_slice/0/vacuum_toroidal_field/r0"]                                     = r0 
    entry["core_profiles/time_slice/0/vacuum_toroidal_field/r0"]                                   = r0 
    entry["equilibrium/time_slice/0/boundary/elongation"]                                          = data[n:=n+1][1]   #  kappa : plasma elongation
    entry["equilibrium/time_slice/0/boundary/triangularity"]                                       = data[n:=n+1][1]   #  delta : plasma triangularity
    entry["equilibrium/time_slice/0/boundary/indentation"]                                         = data[n:=n+1][1]   #  pindent : plasma indentation
    entry["equilibrium/time_slice/0/global_quantities/volume"]                                     = data[n:=n+1][1]   #  volo : plasma volume, meters**3
    entry["equilibrium/time_slice/0/global_quantities/area"]                                       = data[n:=n+1][1]   #  cxareao : plasma cross-sectional area, meters**2
    b0                                                                                        = data[n:=n+1][1]  #  Btor : vacuum toroidal field at rmajor, tesla
    entry["equilibrium/vacuum_toroidal_field/b0"]                                                  = np.asarray([b0])
    entry["equilibrium/time_slice/0/vacuum_toroidal_field/b0"]                                     = b0
    entry["core_profiles/time_slice/0/vacuum_toroidal_field/b0"]                                   = b0
    entry["core_profiles/time_slice/0/global_quantities/current"]                                  = data[n:=n+1][1]   #  total, ohmic, bootstrap, beam and RF currents, amps
    entry["core_profiles/time_slice/0/global_quantities/current_ohmic"]                            = data[n][2]        #  ohmic current
    entry["core_profiles/time_slice/0/global_quantities/current_bootstrap"]                        = data[n][3]        #  bootstrap current
    entry["core_profiles/time_slice/0/global_quantities/current_non_inductive"]                    = data[n][4] + data[n][5] #  beam +RF current
    entry["core_profiles/time_slice/0/global_quantities/beta_pol"]                                 = data[n:=n+1][1]   #  betap : poloidal beta
    entry["core_profiles/time_slice/0/global_quantities/beta_tor"]                                 = data[n:=n+1][1]   #  beta : toroidal beta
    entry["core_profiles/time_slice/0/global_quantities/li_3"]                                     = data[n:=n+1][1]   #  ali : plasma inductance
    entry["core_profiles/time_slice/0/global_quantities/t_e_peaking"]                              = data[n:=n+1][1]   #  te0 : central electron temperature
    entry["core_profiles/time_slice/0/global_quantities/t_i_average_peaking"]                      = data[n:=n+1][1]   #  ti0 : central ion temperature
    psi                                                                                   = np.asarray(data[n:=n+1][1:])  #  psi on rho grid, volt*second/radian
    rho_tor                                                                               = np.asarray(data[n:=n+1][1:])  #  rho grid, meters
    psi_axis                                                                                  = psi[0]
    psi_boundary                                                                              = psi[-1]
    rho_tor_boundary                                                                          = rho_tor[-1]
    entry["core_profiles/time_slice/0/profiles_1d/grid/psi"]                                       = psi 
    entry["core_profiles/time_slice/0/profiles_1d/grid/rho_tor"]                                   = rho_tor
    entry["core_profiles/time_slice/0/profiles_1d/grid/psi_magnetic_axis"]                         = psi_axis
    entry["core_profiles/time_slice/0/profiles_1d/grid/psi_boundary"]                              = psi_boundary 
    entry["core_profiles/time_slice/0/profiles_1d/grid/rho_tor_boundary"]                          = rho_tor_boundary
    entry["core_profiles/time_slice/0/profiles_1d/grid/psi_norm"]                                  = (psi -psi_axis)/(psi_boundary-psi_axis)
    entry["core_profiles/time_slice/0/profiles_1d/grid/rho_tor_norm"]                              = rho_tor/rho_tor_boundary
    entry["core_profiles/time_slice/0/profiles_1d/fcap"]                                           = np.asarray(data[n:=n+1][1:])  #  fcap, (i.e., f(psilim)/f(psi))
    entry["core_profiles/time_slice/0/profiles_1d/gcap"]                                           = np.asarray(data[n:=n+1][1:])  #  gcap, (i.e., <(grad rho)**2*(R0/R)**2>)
    entry["core_profiles/time_slice/0/profiles_1d/hcap"]                                           = np.asarray(data[n:=n+1][1:])  #  hcap, (i.e., (dvolume/drho)/(4*pi*pi*R0*rho))
    entry["core_profiles/time_slice/0/profiles_1d/electrons/temperature"]                          = Te = np.asarray(data[n:=n+1][1:])*1000.0  #  electron temperature, eV
    entry["core_profiles/time_slice/0/profiles_1d/t_i_average"]                                    = Ti = np.asarray(data[n:=n+1][1:])*1000.0  #  ion temperatue, eV
    entry["core_profiles/time_slice/0/profiles_1d/q"]                                              = np.asarray(data[n:=n+1][1:])  # q (i.e., safety factor) profile
    entry["core_profiles/time_slice/0/profiles_1d/electrons/density"]                              = np.asarray(data[n:=n+1][1:])  # electron density, #/meter**3


    for i_prim in range(nprim):        
        
        entry[f"core_profiles/time_slice/0/profiles_1d/ion/{i_prim}/@name"]                        = namep[i_prim].capitalize()               # ion name  
        entry[f"core_profiles/time_slice/0/profiles_1d/ion/{i_prim}/density"]                      = np.asarray(data[n:=n+1][1:]) # primary ion density, #/meter**3, species: d       
        entry[f"core_profiles/time_slice/0/profiles_1d/ion/{i_prim}/temperature"]                  = Ti
    for i_prim in range(nprim):
        entry[f"core_sources/source/ionisation/time_slice/0/profiles_1d/ion/{i_prim}/density"]     = np.asarray(data[n:=n+1][1:]) # ionisation
        entry[f"core_sources/source/recombination/time_slice/0/profiles_1d/ion/{i_prim}/density"]  = np.asarray(data[n:=n+1][1:]) # recombination
        entry[f"core_sources/source/equipartition/time_slice/0/profiles_1d/ion/{i_prim}/density"]  = np.asarray(data[n:=n+1][1:]) # collisional_equipartition ,scx : s
        entry[f"core_sources/source/nbi/time_slice/0/profiles_1d/ion/{i_prim}/density"]            = np.asarray(data[n:=n+1][1:]) # nbi ,sbcx : sink due to cx with be
        entry[f"core_sources/source/total/time_slice/0/profiles_1d/ion/{i_prim}/density"]          = np.asarray(data[n:=n+1][1:]) # total source rate,
        entry[f"core_sources/source/custom_1/time_slice/0/profiles_1d/ion/{i_prim}/density"]       = np.asarray(data[n:=n+1][1:]) # dudt : s dot, #/(meter**3*second)


    for i_prim in range(nprim): 
        entry[f"core_profiles/time_slice/0/profiles_1d/ion/{i_prim}/density_fast"]                 = np.asarray(data[n:=n+1][1:]) # fast ion density, #/meter**3, species: d       

    for i_neu in range(nneu):
        entry[f"core_profiles/time_slice/0/profiles_1d/neutral/{i_neu}/density"]                   = np.asarray(data[n:=n+1][1:]) #  neutral density, #/meter**3, species: d 
        entry[f"core_profiles/time_slice/0/profiles_1d/neutral/{i_neu}/density"]                   = np.asarray(data[n:=n+1][1:]) #  neutral density from wall source, #/meter**3, species: d       
        entry[f"core_profiles/time_slice/0/profiles_1d/neutral/{i_neu}/density"]                   = np.asarray(data[n:=n+1][1:]) #  neutral density from volume source, #/meter**3, species: d       
        entry[f"core_sources/source/particles_to_wall/time_slice/0/profiles_1d/neutral/{i_neu}/density"] = np.asarray(data[n:=n+1][1:]) #  volume source of neutrals, #/(meter**3*second), species: d       

    entry["core_sources/source/nbi/time_slice/0/profiles_1d/electrons/particle"]                   = np.asarray(data[n:=n+1][1:]) # beam electron source, #/(meter**3*second)
    entry["core_sources/source/nbi/time_slice/0/profiles_1d/ion/0/particle"]                       = np.asarray(data[n:=n+1][1:]) # beam ion source, #/(meter**3*second)
    entry["core_profiles/time_slice/0/profiles_1d/j_total"]                                        = np.asarray(data[n:=n+1][1:]) #  total current density, amps/meter**2
    entry["core_profiles/time_slice/0/profiles_1d/j_ohmic"]                                        = np.asarray(data[n:=n+1][1:]) # ohmic current density, amps/meter**2
    entry["core_profiles/time_slice/0/profiles_1d/j_bootstrap"]                                    = np.asarray(data[n:=n+1][1:]) # bootstrap current density, amps/meter**2
    entry["core_profiles/time_slice/0/profiles_1d/j_non_inductive"]                                =(np.asarray(data[n:=n+1][1:]) # beam-driven current density, amps/meter**2
                                                                                                    + np.asarray(data[n:=n+1][1:])) # RF current density, amps/meter**2
    entry["core_profiles/time_slice/0/profiles_1d/_rbfgh"]                                         = np.asarray(data[n:=n+1][1:]) # rho*bp0*fcap*gcap*hcap, tesla*meters
    entry["core_profiles/time_slice/0/profiles_1d/zeff"]                                           = np.asarray(data[n:=n+1][1:]) # rho*bp0*fcap*gcap*hcap, tesla*meters
    entry["core_profiles/time_slice/0/profiles_1d/rotation_frequency_tor_sonic"]                   = np.asarray(data[n:=n+1][1:]) # angular rotation speed profile, rad/sec

    entry["core_transport/model/0/time_slice/0/profiles_1d/electrons/particles/d"]                 = np.asarray(data[n:=n+1][1:]) # electron thermal diffusivity, meters**2/sec on half grid
    entry["core_transport/model/0/time_slice/0/profiles_1d/ion/0/particles/d"]                     = np.asarray(data[n:=n+1][1:]) # ion thermal diffusivity, meters**2/second on half grid
    entry["core_transport/model/5/time_slice/0/profiles_1d/ion/0/energy/d"]                        = np.asarray(data[n:=n+1][1:]) # ion neoclassical thermal conductivity, 1/(meter*second), on half grid
    entry["core_transport/model/0/time_slice/0/profiles_1d/electrons/dpdt"]                        = np.asarray(data[n:=n+1][1:]) # wdot, electrons, watts/meter**3
    entry["core_transport/model/0/time_slice/0/profiles_1d/ion/0/dpdt"]                            = np.asarray(data[n:=n+1][1:]) # wdot, ions, watts/meter**3
    entry["core_transport/model/0/time_slice/0/profiles_1d/conductivity_parallel"]                 = np.asarray(data[n:=n+1][1:]) # electron conduction, watts/meter**3
    entry["core_transport/model/0/time_slice/0/profiles_1d/ion/0/conductivity_parallel"]           = np.asarray(data[n:=n+1][1:]) # ion conduction, watts/meter**3
    entry["core_transport/model/0/time_slice/0/profiles_1d/electrons/particles/v"]                 = np.asarray(data[n:=n+1][1:]) # electron convection, watts/meter**3
    entry["core_transport/model/0/time_slice/0/profiles_1d/ion/0/particles/v"]                     = np.asarray(data[n:=n+1][1:]) # ion convection, watts/meter**3

    # Energy Source
    entry["core_sources/source/nbi/time_slice/0/profiles_1d/electrons/energy"]                     = np.asarray(data[n:=n+1][1:]) #  power to elec. from beam, watts/meter**3
    entry["core_sources/source/equipartition/time_slice/0/profiles_1d/electrons/energy"]           = np.asarray(data[n:=n+1][1:]) #  qdelt : electron-ion equilibration, watts/meter**3
    entry["core_sources/source/nbi/time_slice/0/profiles_1d/ion/0/energy"]                         = np.asarray(data[n:=n+1][1:]) #  power to ions from beam, watts/meter**3
    entry["core_sources/source/auxiliary/time_slice/0/profiles_1d/electrons/energy"]               = np.asarray(data[n:=n+1][1:]) #  qrfe, RF electron heating, watts/meter**3
    entry["core_sources/source/auxiliary/time_slice/0/profiles_1d/ion/0/energy"]                   = np.asarray(data[n:=n+1][1:]) #  qrfi, -RF ion heating, watts/meter**3
    entry["core_sources/source/recombination/time_slice/0/profiles_1d/electrons/energy"]           = np.asarray(data[n:=n+1][1:]) #  qione, electron power density due to recombination and impact ionization, watts/meter**3
    entry["core_sources/source/recombination/time_slice/0/profiles_1d/ion/0/energy"]               = np.asarray(data[n:=n+1][1:]) #  qioni, ion power density due to recombination and impact ionization, watts/meter**3
    entry["core_sources/source/charge_exchange/time_slice/0/profiles_1d/ion/0/energy"]             = np.asarray(data[n:=n+1][1:]) #  qcx, ion power density due to neutral-ion charge exchange, watts/meter**3
    entry["core_sources/source/2d/time_slice/0/profiles_1d/electrons/energy"]                      = np.asarray(data[n:=n+1][1:]) #  2d electron heating, watts/meter**3
    entry["core_sources/source/2d/time_slice/0/profiles_1d/ion/0/energy"]                          = np.asarray(data[n:=n+1][1:]) #  2d ion heating, watts/meter**3
    entry["core_sources/source/fusion/time_slice/0/profiles_1d/electrons/energy"]                  = np.asarray(data[n:=n+1][1:]) #  fusion electron heating, watts/meter**3
    entry["core_sources/source/fusion/time_slice/0/profiles_1d/ion/0/energy"]                      = np.asarray(data[n:=n+1][1:]) #  fusion ion heating, watts/meter**3
    entry["core_sources/source/beam/time_slice/0/profiles_1d/electrons/energy"]                    = np.asarray(data[n:=n+1][1:]) #  beam fusion electron heating, watts/meter**3
    entry["core_sources/source/beam/time_slice/0/profiles_1d/ion/0/energy"]                        = np.asarray(data[n:=n+1][1:]) #  beam fusion ion heating, watts/meter**3
    entry["core_sources/source/qmag/time_slice/0/profiles_1d/electrons/energy"]                    = np.asarray(data[n:=n+1][1:]) #  qmag electron heating, watts/meter**3
    entry["core_sources/source/sawtooth/time_slice/0/profiles_1d/electrons/energy"]                = np.asarray(data[n:=n+1][1:]) #  sawtooth electron heating, watts/meter**3
    entry["core_sources/source/sawtooth/time_slice/0/profiles_1d/ion/0/energy"]                    = np.asarray(data[n:=n+1][1:]) #  sawtooth ion  heating, watts/meter**3
    entry["core_sources/source/radiation/time_slice/0/profiles_1d/ion/0/energy"]                   = np.asarray(data[n:=n+1][1:]) #  radiated power density, watts/meter**3
    entry["core_sources/source/ohmic/time_slice/0/profiles_1d/electrons/energy"]                   = np.asarray(data[n:=n+1][1:]) #  (electron) ohmic power density, watts/meter**3

    # flux surface  
    entry["equilibrium/time_slice/0/profiles_1d/major_radius"]                                     = np.asarray(data[n:=n+1][1:]) #  average major radius of each flux surface, meters, evaluated at elevation of magnetic axis
    entry["equilibrium/time_slice/0/profiles_1d/minor_radius"]                                     = np.asarray(data[n:=n+1][1:]) #  average minor radius of each flux surface, meters, evaluated at elevation of magnetic axis
    entry["equilibrium/time_slice/0/profiles_1d/volume"]                                           = np.asarray(data[n:=n+1][1:]) #  volume of each flux surface, meters**3
    entry["equilibrium/time_slice/0/profiles_1d/elongation"]                                       = np.asarray(data[n:=n+1][1:]) #  elongation of each flux surface
    entry["equilibrium/time_slice/0/profiles_1d/triangularity"]                                    = np.asarray(data[n:=n+1][1:]) #  triangularity of each flux surface
    entry["equilibrium/time_slice/0/profiles_1d/indentation"]                                      = np.asarray(data[n:=n+1][1:]) #  indentation of each flux surface
    entry["equilibrium/time_slice/0/profiles_1d/surface"]                                          = np.asarray(data[n:=n+1][1:]) #  surface area of each flux surface, meter**2
    entry["equilibrium/time_slice/0/profiles_1d/Rhr4pp"]                                           = np.asarray(data[n:=n+1][1:]) #  this is 4*pi*pi*R0*hcap*rho*<ABS(grad rho)>; see file "iter1.ps" for explanation
    entry["equilibrium/time_slice/0/profiles_1d/area"]                                             = np.asarray(data[n:=n+1][1:]) #  cross-sectional area of each flux surface, meters**2
    entry["equilibrium/time_slice/0/profiles_1d/gm7"]                                              = np.asarray(data[n:=n+1][1:]) #  flux surface average absolute grad rho
    entry["equilibrium/time_slice/0/profiles_1d/gm3"]                                              = np.asarray(data[n:=n+1][1:]) #  flux surface average (grad rho)**2
        
    nbbdry                                                                                    = data[n:=n+1][1]              #  nplasbdry : number of points on plasma boundary
    entry["equilibrium/time_slice/0/boundary/outline/r"]                                           = np.asarray(data[n:=n+1][1:]) #  r points for plasma boundary, meters
    entry["equilibrium/time_slice/0/boundary/outline/z"]                                           = np.asarray(data[n:=n+1][1:]) #  z points for plasma boundary, meters

    while n < len(data)-1:
        entry[f"core_profiles/time_slice/0/profiles_1d/{data[n:=n+1][0]}"]                         = np.asarray(data[n][1:]) # beam   torque density, nt-m/m**3

    # fmt:on

    return entry


@File.register(["iterdb"])
class ITERDBFile(File):
    """Read iterdb file  (from gacode)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self) -> Entry:
        if self.url.authority:
            raise NotImplementedError(f"{self.url}")

        path = pathlib.Path(self.url.path)

        if path.suffix.lower() in [".nc", ".h5"]:
            data = File(path, mode="r").read().dump()
        else:
            data = sp_read_iterdb_txt(path)

        return sp_to_imas(data)

    def write(self, d, *args, **kwargs):
        raise NotImplementedError(f"TODO: write ITERDB {self.url}")
