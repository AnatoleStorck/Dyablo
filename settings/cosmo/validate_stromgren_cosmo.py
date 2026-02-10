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

    conf = {"bx":bx, "by":by, "bz":bz, "lmin":lmin, "lmax":lmax, "cor_x":cor_x, "cor_y":cor_y, 
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
def compute_ri(confFile, files, H0, Om) :

   # Read ini file
  conf = read_config(confFile)
    
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

############################################################################################
  
xmf_filename = sys.argv[1] #"test_stromgren_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]
if( len(sys.argv) >= 5 ):
  ini_filename = sys.argv[4]
else:
  ini_filename="last.ini"

print("Validate Stromgren sphere cosmo")
print(f'XMF filename : {xmf_filename}')
print(f'ini filename : {ini_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')  

# Open output file
reader = pyablo.XdmfReader()

# List of output files
files = reader.readTimeSeries(xmf_filename)
files = files[1:] # do not keep the first file at iteration 0

# let's define a cosmology
megaparsec = 3.086e22  # Mpc in meters
Om = 0.3
H0 = 70 # Hubble constant in km/s/Mpc
H0 = H0 * 1e3 / megaparsec  # Convert H0 to s-1
astart = 0.0625
t_start = time_at_a(0.0625, H0, Om)/3600./24./365./1e6

# Cpompute ionisation radius
volumes, times = compute_ri(ini_filename, files, H0, Om)
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

if( diff > target_precision ):
  print( f'Precision target not met {diff:.4} > {target_precision:.4}'  )
  exit(1)
  