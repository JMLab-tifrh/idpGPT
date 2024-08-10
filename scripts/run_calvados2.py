import numpy as np
import openmm as mm
import pickle
import pandas as pd
from MDAnalysis import Universe, Writer
from sys import stdout
import os
from glob import glob
import time
import argparse

##======================== functions ========================##
def getIonicStrength(concentrations, charges):
    """
    Calculate the ionic strength of a solution.

    Args:
    - concentrations (list): List of molar concentrations of ions.
    - charges (list): List of charges of ions.

    Returns:
    - ionic_strength (float): The calculated ionic strength.
    """
    ionic_strength = 0
    for concentration, charge in zip(concentrations, charges):
        ionic_strength += 0.5 * concentration * charge ** 2
    return ionic_strength


# eps0 ---> permittivity of vacuum
fepsw = lambda T, eps0=8.854188 : eps0*(5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3)

# argument parser
def define_args():
    parser = argparse.ArgumentParser(description="run slab simulations from a starting configuration")
    parser.add_argument("-f", "--file", metavar="", type=str, help="initial configuration. [PDB format]")
    parser.add_argument("-o", "--output", metavar="", type=str, help="output filename [XTC/DCD format]")
    parser.add_argument("-s", "--single", metavar="", type=str,
                        help="CG configuration of a single chain [PDB format]")
    parser.add_argument("-n", "--nsteps", metavar="", type=int, help="number of steps to run [int]")
    parser.add_argument("-temp", "--temp", metavar="", type=float, help="temperature in Kelvin [float]")
    parser.add_argument("-freq", "--freq", metavar="", type=int, help="output frequency in steps [int]")
    parser.add_argument("-cn", "--concn", metavar="", type=float, help="anion concentration in M [float]")
    parser.add_argument("-cp", "--concp", metavar="", type=float, help="cation concentration in M [float]")
    parser.add_argument("-nc", "--nc", metavar="", type=float, help="charge of the anion [float]")
    parser.add_argument("-pc", "--pc", metavar="", type=float, help="charge of the cation [float]")
    parser.add_argument("-dt", "--dt", metavar="", type=float, help="time step in ps [float]", default=0.01)
    parser.add_argument("-fric", "--fric", metavar="", type=float,
                        help="friction coefficient [float]", default=1.0)

    return parser
##======================== functions ========================##

##=================== reading user arguments  ===================##
parser = define_args()
args = parser.parse_args()

if not any(vars(args).values()):
    parser.print_help()
    exit(0)

required_args = ['file', 'output', 'single', 'nsteps', 'temp', 'freq', 'concn', 'concp', 'nc', 'pc']

for arg_name in required_args:
    if not getattr(args, arg_name):
        print(f"Error: Argument {arg_name} is required.")
        parser.print_help()
        exit(0)

# Storing arguments into variables
init_config = args.file
output_filename = args.output
single_chain_config = args.single
num_steps = args.nsteps
temperature = args.temp
output_frequency = args.freq
anion_concentration = args.concn
cation_concentration = args.concp
anion_charge = args.nc
cation_charge = args.pc
time_step = args.dt
friction = args.fric

# print info
if init_config:
    print("Initial configuration file:", init_config)
if output_filename:
    print("Output filename:", output_filename)
if single_chain_config:
    print("CG configuration of a single chain:", single_chain_config)
if num_steps:
    print("Number of steps to run:", num_steps)
if temperature:
    print("Temperature:", temperature, "K")
if output_frequency:
    print("Output frequency:", output_frequency)
if anion_concentration:
    print("Anion concentration:", anion_concentration, "M")
if cation_concentration:
    print("Cation concentration:", cation_concentration, "M")
if anion_charge:
    print("Charge of the anion:", anion_charge)
if cation_charge:
    print("Charge of the cation:", cation_charge)
print("time step:", time_step, "ps")
print("friction coefficient:", friction, "1/ps")
##=================== reading user arguments  ===================##

##=================== defining CALVADOS force-field  ===================##
ff = pd.DataFrame({ "one":['R', 'D', 'N', 'E', 'K', 'H', 'Q', 'S', 'C', 'G',
                           'T', 'A', 'M', 'Y', 'V', 'W', 'L', 'I', 'P', 'F'],  
                    "three":['ARG', 'ASP', 'ASN', 'GLU', 'LYS', 'HIS', 'GLN', 'SER', 'CYS', 'GLY',
                             'THR', 'ALA', 'MET', 'TYR', 'VAL', 'TRP', 'LEU', 'ILE', 'PRO', 'PHE'],  
                    "MW":[156.19, 115.09, 114.1, 129.11, 128.17, 137.14, 128.13, 87.08, 103.14, 57.05,
                          101.11, 71.07, 131.2, 163.18, 99.13, 186.22, 113.16, 113.16, 97.12, 147.18],  
                    "lambdas":[0.7307624767517166, 0.0416040480605567, 0.4255859009787713, 
                               0.0006935460962935, 0.1790211738990582, 0.4663667290557992,
                               0.3934318551056041, 0.4625416811611541, 0.5615435099141777,
                               0.7058843733666401, 0.3713162976273964, 0.2743297969040348,
                               0.5308481134337497, 0.9774611449343455, 0.2083769608174481,
                               0.9893764740371644, 0.6440005007782226, 0.5423623610671892,
                               0.3593126576364644, 0.8672358982062975],  
                    "sigmas":[0.6559999999999999, 0.5579999999999999, 0.568, 0.5920000000000001, 0.636,
                              0.608, 0.602, 0.518, 0.5479999999999999, 0.45, 0.562, 0.504, 0.618,
                              0.6459999999999999, 0.5860000000000001, 0.6779999999999999, 0.618,
                              0.618, 0.5559999999999999, 0.636],  
                    "q":[1, -1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                    "CALVADOS1":[0.7249915947715212, 0.0291821237763497, 0.4383272997027284,
                                 0.0061002816086497, 0.0586171731586979, 0.4651948082346978,
                                 0.3268188050525212, 0.4648570130065605, 0.610362354303913,
                                 0.7012713677972457, 0.5379777613307019, 0.0011162643859539,
                                 0.7458993420826714, 0.9950108229594324, 0.4185006852559869,
                                 0.9844235478393932, 0.5563020305733198, 0.6075268330845265,
                                 0.3729641853599348, 0.9216959832175944],  
                    "CALVADOS2":[0.7307624767517166, 0.0416040480605567, 0.4255859009787713,
                                 0.0006935460962935, 0.1790211738990582, 0.4663667290557992,
                                 0.3934318551056041, 0.4625416811611541, 0.5615435099141777,
                                 0.7058843733666401, 0.3713162976273964, 0.2743297969040348,
                                 0.5308481134337497, 0.9774611449343455, 0.2083769608174481,
                                 0.9893764740371644, 0.6440005007782226, 0.5423623610671892,
                                 0.3593126576364644, 0.8672358982062975]})
##=================== defining CALVADOS force-field  ===================##

##===================  building the force field from readable parameters ===================##
# ff_ref = pd.read_csv("calvados.csv")
ff = ff.set_index("three")
keys = list(ff.index)
M = len(keys)
types = {k:i for i, k in enumerate(keys)}

comb_lambda = np.zeros((M, M))
comb_sigma = np.zeros((M, M))
comb_charged = np.zeros((M, M))

for i in range(M):
    for j in range(M):
        comb_lambda[i, j] = 0.5*(ff.loc[keys[i]]["lambdas"] + ff.loc[keys[j]]["lambdas"])
        comb_sigma[i, j] = 0.5*(ff.loc[keys[i]]["sigmas"] + ff.loc[keys[j]]["sigmas"])
        comb_charged[i, j] = ff.loc[keys[i]]["q"]*ff.loc[keys[j]]["q"]
##===================  building the force field from readable parameters ===================##

##=================== system information ===================##
pdbfile = init_config
pdb = Universe(pdbfile)
box = pdb.trajectory[0]._unitcell[:3]*0.1

N = pdb.select_atoms("all").n_atoms
Nres = Universe(single_chain_config).select_atoms("all").n_atoms
Nchains = N//Nres
residues = pdb.select_atoms("all").resnames
NA = 6.022e23
amu = 1.008/NA
##=================== system information ===================#

##=================== system setup ===================#
T = temperature # temperature in Kelvin
RT = 8.3145*T*1e-3
epsw = fepsw(T)  # effective dielectric constant at T
e2 = 1.6021766**2 # square of an unit electric charge
B = e2/(4*np.pi*epsw)*6.022*1000/RT # Bjerrum length
ionic_strength = getIonicStrength([cation_concentration, anion_concentration],
                                  [cation_charge, anion_charge]) # conc in (M) and respective charges
rcut = 4 #nm
## Debye Huckel electrostatics setup
yukawa_kappa = np.sqrt(8*np.pi*B*ionic_strength*6.022/10)
yukawa = mm.CustomNonbondedForce(f"Q*{e2}*exp(-r/{yukawa_kappa})/(4*{np.pi}*{epsw}*r); Q=prod(type1, type2)")
yukawa.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
yukawa.setCutoffDistance(rcut)
yukawa.addTabulatedFunction("prod", mm.Discrete2DFunction(M, M, comb_charged.flatten().tolist()))
yukawa.addPerParticleParameter("type")

## Non bonded interactions setup
two_six = 2**(1/6)
ashbaugh_hatch = mm.CustomNonbondedForce(f"((4*0.8368*((sig/r)^12 - (sig/r)^6)) + (1 - lamb)*0.8368)*step({two_six:.6f}*sig - r) + lamb*(4*0.8368*((sig/r)^12 - (sig/r)^6))*step(r - {two_six:.6f}*sig); sig=sigma(type1, type2); lamb=lambda(type1, type2)")
ashbaugh_hatch.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
ashbaugh_hatch.setCutoffDistance(rcut)
ashbaugh_hatch.addTabulatedFunction('sigma', mm.Discrete2DFunction(M, M, comb_sigma.flatten().tolist()))
ashbaugh_hatch.addTabulatedFunction('lambda', mm.Discrete2DFunction(M, M, comb_lambda.flatten().tolist()))
ashbaugh_hatch.addPerParticleParameter('type')

## reading configuration
pdb = mm.app.PDBFile(init_config)
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2]))

## adding bonds
bonds = mm.HarmonicBondForce()
for c in range(Nchains):
    for i in range(c*Nres, (c+1)*Nres-1):
        bonds.addBond(i, i+1, 0.38, 8033)

## adding forces to calculate
for i in range(N):
    system.addParticle(ff.loc[residues[i]]["MW"])
    ashbaugh_hatch.addParticle([types[residues[i]]])
    yukawa.addParticle([types[residues[i]]])

## adding a COM motion remover
com_motion_remover = mm.openmm.CMMotionRemover(10)

system.addForce(bonds)
system.addForce(ashbaugh_hatch)
system.addForce(yukawa)
system.addForce(com_motion_remover)
##=================== system setup ===================#

##===================== running simulations =====================##
integrator = mm.LangevinMiddleIntegrator(T, friction, time_step) # temp, frictionCoeff, stepsize
simulation = mm.app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
nsteps = num_steps
outfile = output_filename

if len(glob(outfile)):
    os.system(f"rm {outfile}")    

if "xtc" == outfile[-3:]:
    simulation.reporters.append(mm.app.XTCReporter(outfile, output_frequency,
                                                   append=False, enforcePeriodicBox=True))
elif "dcd" == outfile[-3:]:
    simulation.reporters.append(mm.app.DCDReporter(outfile, output_frequency,
                                                   append=False, enforcePeriodicBox=True))
else:
    print("Wrong output format.")
    parser.print_help()
    exit(0)

simulation.reporters.append(mm.app.StateDataReporter(stdout, output_frequency, step=True,
        potentialEnergy=True, temperature=True))
simulation.reporters.append(mm.app.StateDataReporter(outfile[:-4]+".nrg", output_frequency, step=True,
        potentialEnergy=True, temperature=True))

t0 = time.time()
simulation.step(nsteps)
t1 = time.time()
dt = (t1 - t0)
if dt <= 60:
    print(f"Total time taken = {dt:.2f} s")
elif dt > 60 and dt <= 3600:
    print(f"Total time taken = {dt/60:.2f} minutes")
elif dt > 3600 and dt < 86400:
    print(f"Total time taken = {dt/3600:.2f} hrs")
else:
    print(f"Total time taken = {dt/86400:.2f} days")
print(f"Performance = {0.01*nsteps*1e-3*86400/dt:.2f} ns/day")
##===================== running simulations =====================##
