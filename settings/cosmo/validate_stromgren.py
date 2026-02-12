import pyablo, h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as spi
import sys
from configparser import ConfigParser

matplotlib.use("Agg")


### Read ini file and build configuration
def read_config(filename) :
  
    config = ConfigParser(inline_comment_prefixes=('#',';'))
    config.read(filename)
    print('Reading : ', filename)

    # Calculating domain size from the data
    bx = config.getint('amr', 'bx')
    by = config.getint('amr', 'by')
    bz = config.getint('amr', 'bz')

    lmin = config.getint('amr', 'level_min')
    lmax = config.getint('amr', 'level_max')

    cor_x = config.getint('amr', 'coarse_oct_resolution_x')
    cor_y = config.getint('amr', 'coarse_oct_resolution_y')
    cor_z = config.getint('amr', 'coarse_oct_resolution_z')

    xmin = config.getfloat('mesh', 'xmin')
    xmax = config.getfloat('mesh', 'xmax')
    ymin = config.getfloat('mesh', 'ymin')
    ymax = config.getfloat('mesh', 'ymax')
    zmin = config.getfloat('mesh', 'zmin')
    zmax = config.getfloat('mesh', 'zmax')

    # Number of cells
    Nx = bx*cor_x
    Ny = by*cor_y
    Nz = bz*cor_z

    dz = (zmax-zmin)/Nz
    
    cosmo_active = config.getboolean('cosmology', 'active')

    conf = {"cosmo_active":cosmo_active, "bx":bx, "by":by, "bz":bz, "lmin":lmin, "lmax":lmax, "cor_x":cor_x, "cor_y":cor_y, 
            "cor_z":cor_z, "xmin":xmin, "xmax":xmax, "ymin":ymin, "ymax":ymax, "zmin":zmin, "zmax":zmax, "Nx":Nx, "Ny":Ny, "Nz":Nz, "dz":dz}

    return conf
  
def dadt(a, H0, Om):  
    """
    Function to calculate the derivative of the scale factor with respect to time.
    """
    return H0 * np.sqrt(Om / a + (1-Om) * a**2)

def time_at_a(a, H0, Om):
    """
    Function to calculate the time at a given scale factor.
    """
    integral, _ = spi.quad(lambda x: 1 / dadt(x, H0, Om), 0, a)
    return integral
  

### Compute ionisation radius
def compute_ri(conf, files) :
    
  times = []
  radii = []

  for f in files:
          
    with h5py.File(f.replace('xmf', 'h5'), 'r') as h5:
        xe = np.array(h5['rho_HII'])/ np.array(h5['rho'])
        time = h5['scalar_data'].attrs['time']
        
    v = np.sum(xe)*conf['dz']*conf['dz']*conf['dz']

    # Store values and iterations numbers
    radii.append((v*3./4./np.pi)**(1./3.))
    times.append(time)
    
  return np.array(radii), np.array(times)

### Compute ionisation radius
def compute_ri_cosmo(conf, files, H0, Om) :

  times = []
  volumes = []

  for f in files:
          
    with h5py.File(f.replace('xmf', 'h5'), 'r') as h5:
        xe = np.array(h5['rho_HII'])/ np.array(h5['rho'])
        aexp = h5['scalar_data'].attrs['aexp']
        
    v = np.sum(xe)*conf['dz']*conf['dz']*conf['dz']

    # Store values and iterations numbers
    volumes.append(v)
    t = time_at_a(aexp, H0, Om)/3600./24./365./1e6
    times.append(t)
    
  return np.array(volumes), np.array(times)


##############################################################################################
  
xmf_filename = sys.argv[1] #"test_stromgren_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]
if( len(sys.argv) >= 5 ):
  ini_filename = sys.argv[4]
else:
  ini_filename="last.ini"

print("Validate Stromgren sphere")
print(f'XMF filename : {xmf_filename}')
print(f'ini filename : {ini_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')  

# Open output file
reader = pyablo.XdmfReader()

# Read ini file
conf = read_config(ini_filename)

# List of output files
files = reader.readTimeSeries(xmf_filename)
files = files[1:] # do not keep the first file at iteration 0

# Cosmo case
if conf["cosmo_active"] :
  
  # let's define a cosmology
  megaparsec = 3.086e22  # Mpc in meters
  Om = 0.3
  H0 = 70 # Hubble constant in km/s/Mpc
  H0 = H0 * 1e3 / megaparsec  # Convert H0 to s-1
  astart = 0.0625
  t_start = time_at_a(0.0625, H0, Om)/3600./24./365./1e6

  # Cpompute ionisation radius
  volumes, times = compute_ri_cosmo(conf, files, H0, Om)
  times = times - t_start

  plt.figure(figsize=(10,6))
  rmax = 3.341262
  vmax = (4/3)*np.pi*rmax**3
  diff = 1.0 - volumes[-1]/vmax

  plt.figure(figsize=(10,6))
  plt.plot(times, volumes/vmax)
  plt.xlabel("Time [Myr]")
  plt.ylabel("V / Vmax")

  plt.title(f'L1 error = {diff:.4}')
  plt.suptitle('Stromgren sphere')
  plt.grid('.')
  plt.savefig(f'{png_filename}')

  print( f'Exported {png_filename}' )
  print( f'Difference = {diff:.4}'  )
  
else :
  
  # Compute ionisation radius
  ri, times = compute_ri(conf, files)

  rs = 5.4  # kpc
  trec = 122.4 # Myr
  dt = 0.403895 # comes from simulation

  # Theoritical curve
  time = np.arange(0.0, 500.0, 1)/trec
  ri_theo = rs*np.power( 1-np.exp(-time), 1/3 )

  # Compute difference between theory and simulation at 500 Myr
  diff = abs(ri[-1]/rs - ri_theo[-1]/rs) 

  plt.figure(figsize=(10,6))
  plt.plot(times/trec, ri/rs, label=r"size=$16^{3}$   cfrac=0.001") 
  plt.plot(time, ri_theo/rs, label="Theoritical curve")

  # Plot points for different codes at 500 Myr.
  # The data come from https://astronomy.sussex.ac.uk/~iti20/RT_comparison_project/tests1-4.html
  size = 10
  plt.scatter(500/122.4, 1.04, label='c2ray', color='black', s=size)
  plt.scatter(500/122.4, 1.038, label='crash', color='red', s=size)
  plt.scatter(500/122.4, 1.034, label='flash', color='blue', s=size)
  plt.scatter(500/122.4, 1.036, label='ftte', color='green', s=size)
  plt.scatter(500/122.4, 1.007, label='ift', color='orange', s=size)
  plt.scatter(500/122.4, 0.9774, label='otvet', color='purple', s=size)
  plt.scatter(500/122.4, 1.036, label='rsph', color='brown', s=size)
  plt.scatter(500/122.4, 1.042, label='simplex', color='pink', s=size)
  plt.scatter(500/122.4, 1.041, label='zeus', color='cyan', s=size)

  plt.xlabel("time/trec")
  plt.ylabel("rI/rs")
  plt.title(f'L1 error = {diff:.4}')
  plt.suptitle('Stromgren sphere')
  plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,  loc='lower right')
  plt.grid('.')
  plt.savefig(f'{png_filename}')

  print( f'Exported {png_filename}' )
  print( f'Theoritical ri = {ri_theo[-1]/rs:.4}'  )
  print( f'Simulated ri = {ri[-1]/rs:.4}'  )
  print( f'Difference = {diff:.4}'  )

if( diff > target_precision ):
  print( f'Precision target not met {diff:.4} > {target_precision:.4}'  )
  exit(1)
  