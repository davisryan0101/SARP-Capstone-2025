import numpy as np
import scipy
import matplotlib.pyplot as plt
from pygasflow import isentropic_solver
from CoolProp.CoolProp import PropsSI, AbstractState
from pyfluids import Mixture, FluidsList, Input
import scipy.optimize

# Define thermal conductivity values for materials (in W/m·K)
thermal_conductivity = {
    "copper": 398,      # in W/m·K
    "stainless_steel": 16,  # in W/m·K
    "inconel": 11      # in W/m·K
}

# Engine Geometry (unchanged)
chamberLength = 0.06667 # [m]
chamberRadius = 0.01506 # [m]
lengthToThroat = 0.07898 # [m]
throatRadius = 0.00275 # [m]
throatRadiusOfCurvature = 0.01 # [m]
engineLength = 0.22464 # [m]
nozzleExitRadius = 0.04178 # [m]

# Material choice for inner jacket thermal conductivity
material_choice = "copper"  # Change this to "stainless_steel" or "inconel" for other materials

# Set inner jacket thermal conductivity based on material choice
innerJacketThermalConductivity = thermal_conductivity.get(material_choice, 45)  # Default to 45 if invalid material

# Other parameters (unchanged)
innerJacketWallThickness = 0.002 # [m]
gasSpecificHeatRatio = 1.2
gasPrandtlNumber = 0.47
gasViscosity = 0.00009129 # [Pa*s]
gasHeatCapacity = 2023.8 # [J/kg/K] from CEA
gasTemperature = 1370.2 # [K] from CEA
chamberPressure = 500 * 6895 # [Pa]
cstar = 1222.22 # [m/s]
gasMolecularWeight = (28.01 * 0.236) + (44.01 * 0.087) + (2.0157 * 0.015) + (18.02 * 0.232) + (28.02 * 0.425)

# Coolant (unchanged)
coolant = 'Ethanol'
totalCoolantFlowRate = 0.03 # [kg/s]
numberCoolingChannels = 25
coolantChannelFlowRate = totalCoolantFlowRate / numberCoolingChannels # [kg/s]
initialCoolantPressure = 501 / pascalToPsi # [Pa]
initialCoolantTemperature = 290 # [K]

# Channel Geometry (unchanged)
channelWidth = 0.0005 # [m]
channelDepth = 0.005 # [m]
channelAbsoluteRoughness = 0.000045 # [m] carbon steel

# Adjustments for various channel and cooling properties (same as before)
# (include your existing functions here as before...)

# Code for simulation and heat flux calculation remains the same...

# Generate results for each material and store the data
materials = ["copper", "stainless_steel", "inconel"]
results = {}

for material in materials:
    innerJacketThermalConductivity = thermal_conductivity[material]
    
    # Step through channels starting from nozzle exit and calculate properties
    # (Assuming `getCoolantSegmentProperties`, `getRecoveryTemperature`, etc., are defined in the same way)
    axialSegmentLength = engineLength / numberChannelSegments # [m]
    axialPositions = np.zeros(numberChannelSegments) # [m]
    for i in range(len(axialPositions)-1):
        axialPositions[i+1] = axialPositions[i] + axialSegmentLength

    gasSideWallTemp = np.zeros(len(axialPositions)) # [K]
    coolantSideWallTemp = np.zeros(len(axialPositions)) # [K]
    coolantTemperature = np.zeros(len(axialPositions)) # [K]
    heatFlux = np.zeros(len(axialPositions)) # [W/m2/K]
    coolantPressure = np.zeros(len(axialPositions)) # [Pa]
    coolantTemperature[0] = initialCoolantTemperature # [K]
    coolantPressure[0] = initialCoolantPressure # [Pa]

    for i in range(numberChannelSegments-1):
        props = getCoolantSegmentProperties(axialPositions[i], axialPositions[i+1], coolantTemperature[i], coolantPressure[i])
        coolantVaporPressure = props[5]
        if(coolantPressure[i] < coolantVaporPressure):
            print(f"COOLANT BOILING for {material}")
            break
        gasSideWallTemp[i] = props[0] # [K]
        coolantSideWallTemp[i] = props[1] # [K]
        coolantTemperature[i+1] = props[2] # [K]
        heatFlux[i] = props[3] # [W/m2/K]
        coolantPressure[i+1] = props[4] # [Pa]

    results[material] = {
        "zpoints": axialPositions,
        "gasSideWallTemp": gasSideWallTemp,
        "coolantSideWallTemp": coolantSideWallTemp,
        "coolantTemperature": coolantTemperature,
        "heatFlux": heatFlux
    }

# Plot results for each material
plt.figure(figsize=(10, 6))
for material in materials:
    plt.plot(results[material]["zpoints"], results[material]["gasSideWallTemp"], label=f"Gas Side Temp ({material})")
    plt.plot(results[material]["zpoints"], results[material]["coolantSideWallTemp"], label=f"Coolant Side Temp ({material})")

plt.xlabel('Axial Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profiles for Different Materials')
plt.legend()
plt.grid(True)
plt.show()
