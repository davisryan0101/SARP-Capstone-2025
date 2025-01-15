import numpy as np
import scipy
import matplotlib.pyplot as plt
from pygasflow import isentropic_solver
from CoolProp.CoolProp import PropsSI, AbstractState
from pyfluids import Mixture, FluidsList, Input
import scipy.optimize

# Initial Main Engine Cooling Analysis Script
# 
# Process from Heister pg. 211 and Huzel & Huang

# Unit Conversions
metersToFeet = 3.281
metersToInches = 39.37
pascalToPsi = 1 / 6895
kelvinToRankine = 1.8
metersPerSecondToFeetPerSecond = 3.281
btuPerSquareInchSecondFahrenheitToWattsPerSquareMeterKelvin = 2943.6
pascalSecondToPoundPerInchSecond = 0.055997
JoulePerKilogramKelvinToBtuPerPoundFahrenheit = 1 / 4186.8

# Engine Geometry
chamberLength = 0.06667 # [m]
chamberRadius = 0.01506 # [m]
lengthToThroat = 0.07898 # [m]
throatRadius = 0.00275 # [m]
throatRadiusOfCurvature = 0.01 # [m] what is a good value for this?
engineLength = 0.22464 # [m]
nozzleExitRadius = 0.04178 # [m]

# Engine Jacket Material
library_thermal_conductivity = {
    "copper C10100": 398,      # in W/m·K
    "stainless_steel 304/316": 16,  # in W/m·K
    "stainless_steel AISI 1045": 45,   # in W/m·K
    "inconel 718": 11,      # in W/m·K
    "aluminum 6061": 167,   # in W/m·K
    "aluminum 7075": 130,   # in W/m·K
    "aluminum 1050": 205,     # in W/m·K
    "stainless_steel 310": 24  # in W/m·K
}

material_choice = "copper C10100"  # Change this to any other material from the library

# Inner Jacket
innerJacketThermalConductivity = library_thermal_conductivity.get(material_choice, 45)  # Default to 45 if invalid material
innerJacketWallThickness = 0.002 # [m]

# Combustion Gas
gasSpecificHeatRatio = 1.2 # update with CEA values!
gasPrandtlNumber = 0.47 # update with CEA values!
gasViscosity = 0.00009129 # [Pa*s]
gasHeatCapacity = 2023.8 # [J/kg/K] from CEA
gasTemperature = 1370.2 # [K] from CEA
chamberPressure = 500 * 6895 # [Pa]
cstar = 1222.22 # [m/s]
gasMolecularWeight = (28.01 * 0.236) + (44.01 * 0.087) + (2.0157 * 0.015) + (18.02 * 0.232) + (28.02 * 0.425) # get better chemical comp. from Lachlan CEA

# Coolant
coolant = 'Ethanol'
totalCoolantFlowRate = 0.03 # [kg/s]
numberCoolingChannels = 25
coolantChannelFlowRate = totalCoolantFlowRate / numberCoolingChannels # [kg/s]
initialCoolantPressure = 501 / pascalToPsi # [Pa]
initialCoolantTemperature = 290 # [K] could vary depending on orbit thermals, but ethanol pretty constant properties for wide temp & pres ranges

# Channel Geometry
channelWidth = 0.0005 # [m]
channelDepth = 0.005 # [m] what is possible with our machining capabilities? what is optimal from an analysis side of things? seems like high aspect ratio (depth>width) is best
channelAbsoluteRoughness = 0.000045 # [m] carbon steel

# check if the given number of channels would result in no land width between channels
if((np.pi * 2 * throatRadius - numberCoolingChannels * channelWidth) < 0): print("too many channels")

# return radius at given z position of the chamber where z=0 is at the nozzle exit
# note that the input parameters like chamberLength and lengthToThroat are measured with z=0 at the start of the chamber not the nozzle exit
# this function transforms the starting location to the nozzle exit
def getEngineRadius(position):
    if (position >= (engineLength - chamberLength)) and (position <= engineLength):
        return chamberRadius
    elif (position >= (engineLength - lengthToThroat)) and (position < (engineLength - chamberLength)):
        return ((chamberRadius-throatRadius) / ((engineLength-chamberLength)-(engineLength-lengthToThroat))) * (position - (engineLength-lengthToThroat)) + throatRadius
    elif (position >= 0) and (position < (engineLength - lengthToThroat)):
        return ((throatRadius-nozzleExitRadius) / ((engineLength-lengthToThroat)-(0))) * (position - (0)) + nozzleExitRadius
    return ValueError

# returns the effective width of the cooling channel used in calculating heat picked up by coolant in each section
# as per Heister pg. 212: "... for milled channels b would be measured from the center of the land on either side of the channel."
def getChannelLandCenteredWidth(position):
    return 2 * np.pi * (getEngineRadius(position) + innerJacketWallThickness) / numberCoolingChannels

# returns the cross sectional area of a single rectangular channel
# width is the length tangential to a circular cross section of the engine
# depth is the radial length from the coolant side wall of the inner jacket outwards
def getChannelCrossSectionalArea(position):
    return channelWidth * channelDepth

# finds the perimeter of an individual channel cross section at a specified z position
# measured from the nozzle exit
def getChannelPerimeter(position):
    return 2 * channelWidth + 2 * channelDepth

# finds the hydraulic diameter of an individual channel at a specified z position
# measured from nozzle exit
def getChannelHydraulicDiameter(position):
    return 4 * getChannelCrossSectionalArea(position) / getChannelPerimeter(position)

# finds the distance between two z points on the engine wall (of same theta position)
# measured from nozzle exit
def getChannelSegmentLength(position1, position2):
    dz = position2 - position1
    dr = getEngineRadius(position2) - getEngineRadius(position1)
    return np.sqrt((dz**2) + (dr**2))

# finds the mach number at a specified z position in the engine
# position measured from nozzle exit
def getMachNumber(position):
    engineArea = np.pi * (getEngineRadius(position) ** 2)
    throatArea = np.pi * (throatRadius**2)
    areaRatio = engineArea / throatArea
    # supersonic
    if position < (engineLength - lengthToThroat):
        params = isentropic_solver('crit_area_super', areaRatio, gasSpecificHeatRatio, True)
        return params['m']
    # subsonic
    elif position > (engineLength - lengthToThroat):
        params = isentropic_solver('crit_area_sub', areaRatio, gasSpecificHeatRatio, True)
        return params['m']
    else:
        return 1
    
# finds the recovery temperature of the combustion gases as a function of z position in the engine
# this is the temperature used for the gas side heat transfer
# not all of the kinetic energy of the flow is converted to thermal energy in the boundary layer
# so the heat transfer will be based on a slightly lower temperature than the stagnation temperature
# the recovery factor accounts for this loss
def getRecoveryTemperature(position):
    recoveryFactor = gasPrandtlNumber ** (1/3)
    recoveryTempFactor = (1 + ((gasSpecificHeatRatio - 1)/2) * recoveryFactor * (getMachNumber(position)**2))
    stagnationTempFactor = (1 + ((gasSpecificHeatRatio - 1)/2) * (getMachNumber(position)**2))
    recoveryStagnationTempRatio = recoveryTempFactor / stagnationTempFactor
    return recoveryStagnationTempRatio * gasTemperature

def getCoolantDensity(pressure, temperature):
    return PropsSI("D", "T", temperature, "P", pressure, coolant)

def getCoolantHeatCapacity(pressure, temperature):
    return PropsSI("C", "T", temperature, "P", pressure, coolant)

def getCoolantViscosity(pressure, temperature):
    return PropsSI("V", "T", temperature, "P", pressure, coolant)

# the colebrook equation is an implicit formula used to calculate the darcy friction factor
# this function is minimized in the getFrictionFactor function to find the friction factor that equates the left and right sides of the equation
def colebrookEquation(frictionFactor, reynoldsNumber, absoluteRoughness, hydraulicDiameter):
    left = 1 / np.sqrt(frictionFactor)
    right = -2 * np.log10(absoluteRoughness / (3.7 * hydraulicDiameter) + 2.51 / (reynoldsNumber * np.sqrt(frictionFactor)))
    return abs(left - right)

# find the darcy friction factor using the colebrook equation for a given reynolds number, absolute roughness, and hydraulic diameter
def getFrictionFactor(reynoldsNumber, absoluteRoughness, hydraulicDiameter):
    frictionFactor = scipy.optimize.minimize_scalar(colebrookEquation, bounds=(0, 1), method='bounded', tol=0.0001, args=(reynoldsNumber, absoluteRoughness, hydraulicDiameter))['x']
    return frictionFactor

# finds the gas side heat transfer coefficient using Bartz Equation
def getGasSideHeatTransferCoefficient(position, gasSideWallTemp):
    stagnationCorrection = 1 + ((gasSpecificHeatRatio - 1)/2) * (getMachNumber(position)**2)
    throatDiameter = 2 * getEngineRadius(position)
    gasNozzleStartTemp = gasTemperature # [K]
    gasNozzleStartPressure = chamberPressure # [Pa]
    gasNozzleStartViscosity = gasViscosity  # [Pa*s]
    gasNozzleStartHeatCapacity =  gasHeatCapacity # [J/kg/K]
    gasNozzleStartPrandtlNumber = gasPrandtlNumber
    throatArea = np.pi * (throatRadius**2) # [m^2]
    engineArea = np.pi * (getEngineRadius(position)**2) # [m^2]
    areaRatio = throatArea / engineArea
    tempRatio = gasSideWallTemp / gasNozzleStartTemp
    sigma = 1 / (((0.5 * tempRatio * stagnationCorrection + 0.5)**(0.68)) * ((stagnationCorrection)**(0.12)))
    bartzHTC = (0.026/(throatDiameter**0.2))*(((gasNozzleStartViscosity**0.2) * gasNozzleStartHeatCapacity)/(gasNozzleStartPrandtlNumber**0.6))*((gasNozzleStartPressure/cstar)**0.8)*((throatDiameter/throatRadiusOfCurvature)**0.1)*(areaRatio**0.9)*sigma
    return bartzHTC

# Seider-Tate correlation
def getCoolantHeatTransferCoefficient(position, velocity, bulkTemperature, wallTemperature, pressure):
    coolantThermalConductivity = PropsSI("L", "T", bulkTemperature, "P", pressure, coolant)
    coolantBulkViscosity = PropsSI("V", "T", bulkTemperature, "P", pressure, coolant)
    if(wallTemperature <= 273): 
        coolantWallViscosity = PropsSI("V", "T", 300, "P", pressure, coolant)
    else: 
        coolantWallViscosity = PropsSI("V", "T", wallTemperature, "P", pressure, coolant)    
    reynoldsNumber = getCoolantDensity(pressure, bulkTemperature) * getChannelHydraulicDiameter(position) * velocity / coolantBulkViscosity
    prandtlNumber = getCoolantHeatCapacity(pressure, bulkTemperature) * coolantBulkViscosity / coolantThermalConductivity
    a = 0.023 # Typically 0.023, but may need adjustment to 0.00805 due to small coolant channel size as per https://www.sciencedirect.com/science/article/pii/0017931094900116
    m = 4/5 # find value for methanol?
    n = 1/3 # find value for methanol?
    b = 0.114 # find value for methanol?
    nusseltNumber = a * (reynoldsNumber ** m) * (prandtlNumber ** n) * ((coolantBulkViscosity / coolantWallViscosity)**b)
    return nusseltNumber * coolantThermalConductivity / getChannelHydraulicDiameter(position)

# finds the gas side wall temperature, coolant side wall temperature, coolant bulk temperature, steady state heat flux, and coolant pressure 
# of a coolant channel segment (increment of z axis length based on numberChannelSegments variable)
def getCoolantSegmentProperties(position, nextPosition, upstreamCoolantTemperature, upstreamCoolantPressure):
    gasRecoveryTemperature = getRecoveryTemperature(position)
    coolantVaporPressure = PropsSI("P", "T", upstreamCoolantTemperature, "Q", 0, coolant)
    coolantVelocity = coolantChannelFlowRate / (getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * getChannelCrossSectionalArea(position)) # [m/s]
    #print("coolant velocity: " + str(coolantVelocity))
    coolantReynoldsNumber = getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * coolantVelocity * getChannelHydraulicDiameter(position) / getCoolantViscosity(upstreamCoolantPressure, upstreamCoolantTemperature)
    gasSideWallTemp = minimizeHeatFluxDifference(position, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure) # [K]
    gasHeatTransferCoefficient = getGasSideHeatTransferCoefficient(position, gasSideWallTemp)
    #print("gas side heat transfer coefficient: " + str(gasHeatTransferCoefficient))
    heatFlux = gasHeatTransferCoefficient * (gasRecoveryTemperature - gasSideWallTemp)
    coolantSideWallTemp = -(heatFlux * innerJacketWallThickness / innerJacketThermalConductivity - gasSideWallTemp)
    coolantTemperature = upstreamCoolantTemperature + (1 / (coolantChannelFlowRate * getCoolantHeatCapacity(upstreamCoolantPressure, upstreamCoolantTemperature))) * (heatFlux * getChannelSegmentLength(position, nextPosition) * getChannelLandCenteredWidth(position)) # [K]
    wallHeatFlux = innerJacketThermalConductivity * (gasSideWallTemp - coolantSideWallTemp) / innerJacketWallThickness
    coolantHeatFlux = getCoolantHeatTransferCoefficient(position, coolantVelocity, upstreamCoolantTemperature, coolantSideWallTemp, upstreamCoolantPressure) * (coolantSideWallTemp - upstreamCoolantTemperature)
    #print("gas side heat flux: " + str(heatFlux))
    #print("wall heat flux: " + str(heatFlux))
    #print("coolant side heat flux: " + str(heatFlux))
    #channelFrictionCoefficient = 0.0014 + 0.125/((getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * coolantVelocity * getChannelHydraulicDiameter(position) / getCoolantViscosity(upstreamCoolantPressure, upstreamCoolantTemperature))**0.32) # UPDATE W/ REAL VALUE
    channelFrictionFactor = getFrictionFactor(coolantReynoldsNumber, channelAbsoluteRoughness, getChannelHydraulicDiameter(position))
    #segmentPressureDrop = channelFrictionFactor * (getChannelSegmentLength(position, nextPosition)/getChannelHydraulicDiameter(position)) * 2 * getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * (coolantVelocity ** 2) # [Pa]
    segmentPressureDrop = channelFrictionFactor * getChannelSegmentLength(position, nextPosition) / getChannelHydraulicDiameter(position) * getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) / 2 * (coolantVelocity ** 2)
    coolantPressure = upstreamCoolantPressure - segmentPressureDrop
    return [gasSideWallTemp, coolantSideWallTemp, coolantTemperature, heatFlux, coolantPressure, coolantVaporPressure]

# calcualte gas side and coolant side heat transfer coefficients and find difference between the two heat fluxes for a set of parameters
# steady state requires the gas side and coolant size heat fluxes to be equal, so this function must be iterated until it returns zero
def heatFluxDifference(gasSideWallTemp, position, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure):
    gsHTC = getGasSideHeatTransferCoefficient(position, gasSideWallTemp) # [W/m2/K]
    gasHeatFlux = gsHTC * (getRecoveryTemperature(position) - gasSideWallTemp) # [W/m2]
    coolantSideWallTemperature = -(gasHeatFlux * innerJacketWallThickness / innerJacketThermalConductivity - gasSideWallTemp) # [K]
    csHTC = getCoolantHeatTransferCoefficient(position, coolantVelocity, upstreamCoolantTemperature, coolantSideWallTemperature, upstreamCoolantPressure) # [W/m2/K]
    coolantHeatFlux = csHTC * (coolantSideWallTemperature - upstreamCoolantTemperature)
    return abs(coolantHeatFlux - gasHeatFlux)

    # fin effects accounted using https://www.researchgate.net/publication/343702059_A_Fin_Analogy_Model_for_Thermal_Analysis_of_Regeneratively_Cooled_Rocket_Engines 
    #m = np.sqrt(csHTC / (innerJacketThermalConductivity * (np.pi * (getChannelHydraulicDiameter(position) + innerJacketWallThickness) / numberCoolingChannels - channelWidth)))
    #finEfficiency = (np.tanh(m * channelDepth)) / (m * channelDepth)
    #wallConductionResistancePerUnitLength = np.log(1 + 2 * innerJacketWallThickness / getChannelHydraulicDiameter(position)) / (2 * np.pi * innerJacketThermalConductivity) # [m*K/W]
    #coolantConvectionResistancePerUnitLength = 1 / (numberCoolingChannels * channelWidth * csHTC) # [m*K/W]
    #coolantFinResistancePerUnitLength = 1 / (2 * finEfficiency * numberCoolingChannels * csHTC * channelDepth) # [m*K/W]
    #totalResistance = wallConductionResistancePerUnitLength + (1 / ((1 / coolantConvectionResistancePerUnitLength) + (1 / coolantFinResistancePerUnitLength))) # [m*K/W]
    #gasHeatPerUnitLength = gasHeatFlux * 2 * np.pi * getEngineRadius(position) / numberCoolingChannels # [W/m]
    #coolantHeatPerUnitLength = 1/totalResistance * (gasSideWallTemp - upstreamCoolantTemperature) # [W/m]
    #return abs(coolantHeatPerUnitLength - gasHeatPerUnitLength)

# iterate gas side and coolant side heat fluxes until they equalize (steady state)
# minimizes the heatFluxDifference function for the gas side wall temperature that equalizes the heat fluxes
def minimizeHeatFluxDifference(position, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure):
    gasSideWallTemp = scipy.optimize.minimize_scalar(heatFluxDifference, bounds=(350, getRecoveryTemperature(position)), method='bounded', tol=0.1, args=(position, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure))['x'] # could experiment with tolerance here
    return gasSideWallTemp

# Step through channels starting from nozzle exit
# Collect heat flux, temperatures, pressures, etc from each segment
# Store properties in vectors to plot against axial position along engine
numberChannelSegments = 250
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
    # check if coolant is boiling, if so end coolant channel step through
    if(coolantPressure[i] < coolantVaporPressure):
        print("COOLANT BOILING")
        break
    gasSideWallTemp[i] = props[0] # [K]
    coolantSideWallTemp[i] = props[1] # [K]
    coolantTemperature[i+1] = props[2] # [K]
    heatFlux[i] = props[3] # [W/m2/K]
    coolantPressure[i+1] = props[4] # [Pa]

# Plot various quantities
zpoints = axialPositions
engineRadius = np.zeros(len(zpoints))
for i in range(len(engineRadius)):
    engineRadius[i] = getEngineRadius(zpoints[i])
landCenteredWidth = np.zeros(len(zpoints))
for i in range(len(landCenteredWidth)):
    landCenteredWidth[i] = getChannelLandCenteredWidth(zpoints[i])
segmentLength = np.zeros(len(zpoints))
for i in range(len(segmentLength)-1):
    segmentLength[i] = getChannelSegmentLength(zpoints[i], zpoints[i+1])
machNumber = np.zeros(len(zpoints))
for i in range(len(machNumber)):
    machNumber[i] = getMachNumber(zpoints[i])
recoveryTemperature = np.zeros(len(zpoints))
for i in range(len(recoveryTemperature)):
    recoveryTemperature[i] = getRecoveryTemperature(zpoints[i])
plt.plot(zpoints, gasSideWallTemp)
plt.plot(zpoints, coolantSideWallTemp)
plt.plot(zpoints, coolantTemperature)
plt.grid
plt.gca().set_xlim([0, 0.25])
plt.show()


# TODO:
# narrow down engine params
# debug fin effects
# add better code to make pretty plots
# comment all functions
# add code evaluating stresses
