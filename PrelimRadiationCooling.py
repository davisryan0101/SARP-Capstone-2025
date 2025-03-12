import numpy as np
import scipy
import matplotlib.pyplot as plt
from pygasflow import isentropic_solver
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI, AbstractState, PT_INPUTS
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

def runCoolingAnalysis(chamberLength, chamberRadius, lengthToThroat, throatRadius, throatRadiusOfCurvature, engineLength, nozzleExitRadius, gasSpecificHeatRatio, gasPrandtlNumber, gasViscosity, gasHeatCapacity, gasTemperature, chamberPressure, cstar, innerJacketElasticModulus, innerJacketThermalExpansion, innerJacketPoissonRatio, jacketEmissivity):

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

    ##########################################################################################################################################################################################################################################################
                # ISENTROPIC RELATIONS #                                                                                                                                                                                                                                sup fucker
    ##########################################################################################################################################################################################################################################################

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

    def getGasPressure(position):
        gasPressure = chamberPressure * (1 + ((gasSpecificHeatRatio - 1) / 2 ) * getMachNumber(position)**2 ) ** (-(gasSpecificHeatRatio / (gasSpecificHeatRatio - 1)))
        return gasPressure

    ##########################################################################################################################################################################################################################################################
                # GAS AND COOLANT PROPERTIES #                                                                                                                                                                                                                                sup fucker
    ##########################################################################################################################################################################################################################################################

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

    # finds the gas side wall temperature, coolant side wall temperature, coolant bulk temperature, steady state heat flux, and coolant pressure 
    # of a coolant channel segment (increment of z axis length based on numberChannelSegments variable)
    def getCoolantSegmentProperties(position):
        gasRecoveryTemperature = getRecoveryTemperature(position)
        gasSideWallTemp = minimizeHeatFluxDifference(position) # [K]
        gasHeatTransferCoefficient = getGasSideHeatTransferCoefficient(position, gasSideWallTemp)
        heatFlux = gasHeatTransferCoefficient * (gasRecoveryTemperature - gasSideWallTemp)
        radiativeHeatFlux = jacketEmissivity * 5.67e-8 * (gasSideWallTemp ** 4)
        #maxJacketStress = ((upstreamCoolantPressure - getGasPressure(position)) * getEngineRadius(position)) / innerJacketWallThickness + (innerJacketElasticModulus * innerJacketThermalExpansion * heatFlux * innerJacketWallThickness) / (2 * (1 - innerJacketPoissonRatio) * innerJacketThermalConductivity)
        return [gasSideWallTemp, heatFlux, radiativeHeatFlux]

    # calcualte gas side and coolant side heat transfer coefficients and find difference between the two heat fluxes for a set of parameters
    # steady state requires the gas side and coolant size heat fluxes to be equal, so this function must be iterated until it returns zero
    def heatFluxDifference(gasSideWallTemp, position):
        gsHTC = getGasSideHeatTransferCoefficient(position, gasSideWallTemp) # [W/m2/K]
        gasHeatFlux = gsHTC * (getRecoveryTemperature(position) - gasSideWallTemp) # [W/m2]
        radiativeHeatFlux = jacketEmissivity * 5.67e-8 * (gasSideWallTemp ** 4)
        return abs(radiativeHeatFlux - gasHeatFlux)

    # iterate gas side and coolant side heat fluxes until they equalize (steady state)
    # minimizes the heatFluxDifference function for the gas side wall temperature that equalizes the heat fluxes
    def minimizeHeatFluxDifference(position):
        gasSideWallTemp = scipy.optimize.minimize_scalar(heatFluxDifference, bounds=(300, getRecoveryTemperature(position)), method='bounded', tol=0.1, args=(position))['x'] # could experiment with tolerance here
        return gasSideWallTemp

    # Step through channels starting from nozzle exit
    # Collect heat flux, temperatures, pressures, etc from each segment
    # Store properties in vectors to plot against axial position along engine
    numberSegments = 500
    axialSegmentLength = engineLength / numberSegments # [m]
    axialPositions = np.zeros(numberSegments) # [m]
    for i in range(len(axialPositions)-1):
        axialPositions[i+1] = axialPositions[i] + axialSegmentLength
    gasSideWallTemp = np.zeros(len(axialPositions)) # [K]
    heatFlux = np.zeros(len(axialPositions)) # [W/m2/K]
    radiativeHeatFlux = np.zeros(len(axialPositions)) # [Pa]
    #maxJacketStress = np.zeros(len(axialPositions)) # [Pa]
    for i in range(numberSegments-1):
        props = getCoolantSegmentProperties(axialPositions[i])
        gasSideWallTemp[i] = props[0] # [K]
        heatFlux[i] = props[1] # [W/m2/K]
        radiativeHeatFlux[i] = props[2] # [W/m2/K]
        #maxJacketStress[i] = props[5] # [Pa]

    # Plot various quantities
    zpoints = axialPositions
    engineRadius = np.zeros(len(zpoints))
    for i in range(len(engineRadius)):
        engineRadius[i] = getEngineRadius(zpoints[i])
    machNumber = np.zeros(len(zpoints))
    for i in range(len(machNumber)):
        machNumber[i] = getMachNumber(zpoints[i])
    recoveryTemperature = np.zeros(len(zpoints))
    for i in range(len(recoveryTemperature)):
        recoveryTemperature[i] = getRecoveryTemperature(zpoints[i])
    gasStaticPressure = np.zeros(len(zpoints))
    for i in range(len(gasStaticPressure)):
        gasStaticPressure[i] = getGasPressure(zpoints[i])

    return [zpoints, engineRadius, machNumber, recoveryTemperature, gasStaticPressure, gasSideWallTemp, heatFlux, radiativeHeatFlux]

##########################################################################################################################################################################################################################################################
            # STRUCTURAL GEOMETRY #                                                                                                                                                                                                                                sup fucker
##########################################################################################################################################################################################################################################################

# Engine Geometry
chamberLength = 0.08305 # [m]
chamberRadius = 0.023725 # [m]
lengthToThroat = 0.11006 # [m]
throatRadius = 0.00839 # [m]
throatRadiusOfCurvature = 0.005 # [m] what is a good value for this?
engineLength = 0.11148+lengthToThroat # [m]
nozzleExitRadius = 0.048905 # [m]

# Inner Jacket
jacketEmissivity = 0.8
innerJacketElasticModulus = 1.2e11 # [Pa] @ 1000 C
innerJacketThermalExpansion = 12.2e-6
innerJacketPoissonRatio = 0.3

##########################################################################################################################################################################################################################################################
            # COMBUSTION GAS DEFINITION #                                                                                                                                                                                                                                sup fucker
##########################################################################################################################################################################################################################################################

gasSpecificHeatRatio = 1.23 # update with CEA values!
gasPrandtlNumber = 0.531 # update with CEA values!
gasViscosity = 0.0000536 # [Pa*s]
gasHeatCapacity = 2462 # [J/kg/K] from CEA
gasTemperature = 1484.46 # [K] from CEA
chamberPressure = 100 * 6895 # [Pa]
cstar = 1339.5 * 0.96 # [m/s]

############
# RUN CODE #
############

[zpoints, engineRadius, machNumber, recoveryTemperature, gasStaticPressure, gasSideWallTemp, heatFlux, radiativeHeatFlux] = runCoolingAnalysis(chamberLength, chamberRadius, lengthToThroat, throatRadius, throatRadiusOfCurvature, engineLength, nozzleExitRadius, gasSpecificHeatRatio, gasPrandtlNumber, gasViscosity, gasHeatCapacity, gasTemperature, chamberPressure, cstar, innerJacketElasticModulus, innerJacketThermalExpansion, innerJacketPoissonRatio, jacketEmissivity)

plt.plot(zpoints, gasSideWallTemp, label="Wall Temp [K]")
#plt.plot(zpoints, heatFlux, label="Convective Heat Flux [W/m^2]")
#plt.plot(zpoints, radiativeHeatFlux, label="Radiative Heat Flux [W/m^2]")
#plt.plot(zpoints, maxJacketStress, label="Max Inner Jacket Stress [Pa]")
plt.grid()
plt.legend()
plt.gca().set_xlabel("Axial (z) Position (from nozzle exit) [m]")
plt.gca().set_ylabel("Temperature [K]")
#plt.gca().set_ylabel("Heat Flux [W/m^2]")
plt.gca().set_xlim([0, 0.25])
plt.show


    # TODO:
    # 