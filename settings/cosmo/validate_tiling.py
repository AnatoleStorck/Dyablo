import pyablo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from matplotlib import colors

matplotlib.use("Agg")

        
### Read ini file and build configuration
def read_config(filename) :
    config = ConfigParser(inline_comment_prefixes=('#',';'))
    config.read(filename)

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

    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Ny
    dz = (zmax-zmin)/Nz

    conf = {"bx":bx, "by":by, "bz":bz, "lmin":lmin, "lmax":lmax, "cor_x":cor_x, "cor_y":cor_y, 
            "cor_z":cor_z, "xmin":xmin, "xmax":xmax, "ymin":ymin, "ymax":ymax, "zmin":zmin, "zmax":zmax, "Nx":Nx, "Ny":Ny, "Nz":Nz, 
            "dx":dx, "dy":dy, "dz":dz}

    return conf


############################################################################################

  
xmf_filename = sys.argv[1] #"tiling_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]
if len(sys.argv) >= 5 :
  ini_filename = sys.argv[4]
else:
  ini_filename="last.ini"

print("Validate tiling")
print(f'XMF filename : {xmf_filename}')
print(f'ini filename : {ini_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')  

# Open output file
reader = pyablo.XdmfReader()

# List of output files
series = reader.readTimeSeries(xmf_filename)
files = series[1:]
print(files[-1])

snap = reader.readSnapshot(files[-1]) 
conf = read_config(ini_filename)

pos = conf["Nz"]//2
mask = snap.getSortingMask3d(conf["lmin"], conf["bx"], conf["by"], conf["bz"], conf["cor_x"], conf["cor_y"], conf["cor_z"])
ext = np.array([conf["xmin"], conf["xmax"], conf["zmin"], conf["zmax"]])
rho = np.array(snap.readAllFloat('rho'))[mask].reshape((conf["Nz"], conf["Ny"], conf["Nx"]))
slice = rho[:,pos,:]

fig = plt.plot(figsize=(10, 6))
p = plt.imshow(slice, extent=ext, cmap='jet', alpha=.8, norm=colors.LogNorm())
plt.xlabel('z [Mpc]')
plt.ylabel('y [Mpc]')
plt.title("Rho tiled")
plt.savefig(f'{png_filename}', dpi=100)

print( f'Exported {png_filename}' )

middle = conf["Nx"]//2
first_row = slice[0]
left = first_row[0:middle]
right = first_row[middle:conf["Nx"]]

# Check if values are the same
np.testing.assert_allclose(left, right, rtol=target_precision, atol=target_precision)

# Check if shape is correct
shape = rho.shape
expected_shape = (conf["Nz"], conf["Ny"], conf["Nz"]*2) # Because rho has been reshaped as (conf["Nz"], conf["Ny"], conf["Nx"])
if shape != expected_shape :
  print( f'Shape of rho array {rho.shape} does not match expected shape {expected_shape}')
  exit(1)
  
# Check if dimensions are correct
if conf["xmax"] != 2*conf["ymax"] and conf["xmax"]!=2*conf["zmax"]:
  print( f'Dimensions are not correct : {conf["xmax"]} should be 2*{conf["ymax"]} and 2*{conf["zmax"]}')
  exit(1)
