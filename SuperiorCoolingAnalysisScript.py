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

def runCoolingAnalysis(chamberLength, chamberRadius, lengthToThroat, throatRadius, throatRadiusOfCurvature, engineLength, nozzleExitRadius, innerJacketThermalConductivity, innerJacketWallThickness, gasSpecificHeatRatio, gasPrandtlNumber, gasViscosity, gasHeatCapacity, gasTemperature, chamberPressure, cstar, gasMolecularWeight, fuel, coolantWaterMassContent, totalCoolantFlowRate, numberCoolingChannels, initialCoolantPressure, initialCoolantTemperature, channelWidth, channelDepthChamber, channelDepthThroat, channelDepthNozzle, channelAbsoluteRoughness, innerJacketElasticModulus, innerJacketThermalExpansion, innerJacketPoissonRatio):

    coolantChannelFlowRate = totalCoolantFlowRate / numberCoolingChannels # [kg/s]
    
    coolantFuelMassContent = 1 - coolantWaterMassContent
    moleEthanol = coolantFuelMassContent / (PropsSI('molar_mass', 'P', 101325, 'T', 298.15, fuel) * 1000)
    moleWater = coolantWaterMassContent / (PropsSI('molar_mass', 'P', 101325, 'T', 298.15, "Water") * 1000)
    totalMoles = moleEthanol + moleWater
    moleFractionEthanol = moleEthanol / totalMoles
    moleFractionWater = moleWater / totalMoles
    coolant = 'HEOS::{}[{}]&Water[{}]'.format(fuel, moleFractionEthanol, moleFractionWater)

    #visc = PropsSI('C','T',349,'P',4692162.001, coolant)
    #waterVisc = PropsSI('V','T',349,'P',4692161.001, "Water")
    #ethVisc = PropsSI('T','Q',0,'P',4692161.001, "Ethanol")
    #print(str(waterVisc))
    #print(str(ethVisc))
    #print(str(visc))

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

    def getChannelDepth(position):
        if (position <= (engineLength - lengthToThroat)):
            channelDepth = ((channelDepthThroat-channelDepthNozzle)/((engineLength - lengthToThroat)-0))*(position-0) + channelDepthNozzle
        elif ((position > (engineLength - lengthToThroat)) and (position <= (engineLength - chamberLength))):
            channelDepth = ((channelDepthChamber-channelDepthThroat)/((engineLength - chamberLength)-(engineLength - lengthToThroat)))*(position-(engineLength - lengthToThroat)) + channelDepthThroat
        else:
            channelDepth = channelDepthChamber
        return channelDepth

    # returns the effective width of the cooling channel used in calculating heat picked up by coolant in each section
    # as per Heister pg. 212: "... for milled channels b would be measured from the center of the land on either side of the channel."
    def getChannelLandCenteredWidth(position):
        return 2 * np.pi * (getEngineRadius(position) + innerJacketWallThickness) / numberCoolingChannels

    # returns the cross sectional area of a single rectangular channel
    # width is the length tangential to a circular cross section of the engine
    # depth is the radial length from the coolant side wall of the inner jacket outwards
    def getChannelCrossSectionalArea(position):
        return channelWidth * getChannelDepth(position)

    # finds the perimeter of an individual channel cross section at a specified z position
    # measured from the nozzle exit
    def getChannelPerimeter(position):
        return 2 * channelWidth + 2 * getChannelDepth(position)

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

    def getCoolantDensity(pressure, temperature):
        fuelDensity = PropsSI("D", "T", temperature, "P", pressure, fuel)
        waterDensity = PropsSI("D", "T", temperature, "P", pressure, "Water")
        return coolantWaterMassContent * waterDensity + (1 - coolantWaterMassContent) * fuelDensity

    def getCoolantHeatCapacity(pressure, temperature):
        fuelHeatCapacity = PropsSI("C", "T", temperature, "P", pressure, fuel)
        waterHeatCapacity = PropsSI("C", "T", temperature, "P", pressure, "Water")
        return coolantWaterMassContent * waterHeatCapacity + (1 - coolantWaterMassContent) * fuelHeatCapacity

    def getCoolantViscosity(pressure, temperature):
        if(temperature < 514):
            saturationPressure = PropsSI("P", "T", temperature, "Q", 0, fuel)
        else:
            saturationPressure = PropsSI("P", "T", 510, "Q", 0, fuel)
        if(abs(pressure - saturationPressure) > 1):
            fuelViscosity = PropsSI("V", "T", temperature, "P", pressure, fuel)
            waterViscosity = PropsSI("V", "T", temperature, "P", pressure, "Water")
        else:
            fuelViscosity = PropsSI("V", "T", temperature-1, "P", pressure, fuel)
            waterViscosity = PropsSI("V", "T", temperature-1, "P", pressure, "Water")
        return coolantWaterMassContent * waterViscosity + (1 - coolantWaterMassContent) * fuelViscosity

    def getCoolantThermalConductivity(pressure, temperature):
        fuelThermalConductivity = PropsSI("L", "T", temperature, "P", pressure, fuel)
        waterThermalConductivity = PropsSI("L", "T", temperature, "P", pressure, "Water")
        return coolantWaterMassContent * waterThermalConductivity + (1 - coolantWaterMassContent) * fuelThermalConductivity

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
        return 1 * bartzHTC

    # Seider-Tate correlation
    def getCoolantHeatTransferCoefficient(position, nextPosition, velocity, bulkTemperature, wallTemperature, pressure):
        #meanTemp = (abs(bulkTemperature) + abs(wallTemperature)) / 2
        if(wallTemperature <= initialCoolantTemperature): 
            coolantWallViscosity = getCoolantViscosity(pressure, initialCoolantTemperature)
            #coolantWallViscosity = PropsSI("V", "T", initialCoolantTemperature, "P", pressure, coolant)
        else: 
            coolantWallViscosity = getCoolantViscosity(pressure, wallTemperature)
            #coolantWallViscosity = PropsSI("V", "T", wallTemperature, "P", pressure, coolant)
        coolantBulkThermalConductivity = getCoolantThermalConductivity(pressure, bulkTemperature)
        #coolantBulkThermalConductivity = PropsSI("L", "T", bulkTemperature, "P", pressure, coolant)
        coolantBulkViscosity = getCoolantViscosity(pressure, bulkTemperature)
        #coolantBulkViscosity = PropsSI("V", "T", bulkTemperature, "P", pressure, coolant)
        coolantBulkDensity = getCoolantDensity(pressure, bulkTemperature)
        coolantBulkHeatCapacity = getCoolantHeatCapacity(pressure, bulkTemperature)
        coolantVaporPres = PropsSI("P", "T", bulkTemperature, "Q", 0, fuel)
        coolantSaturationTemperature = PropsSI("T", "P", pressure, "Q", 0, fuel)
        
        reynoldsNumber = coolantBulkDensity * getChannelHydraulicDiameter(position) * velocity / coolantBulkViscosity
        prandtlNumber = coolantBulkHeatCapacity * coolantBulkViscosity / coolantBulkThermalConductivity
        a = 0.023 # Typically 0.023, but may need adjustment to 0.00805 due to small coolant channel size as per https://www.sciencedirect.com/science/article/pii/0017931094900116
        m = 4/5 # find value for methanol?
        n = 1/3 # find value for methanol?
        b = 0#0.114 # find value for methanol?
        nusseltNumber = a * (reynoldsNumber ** m) * (prandtlNumber ** n) * ((coolantBulkViscosity / coolantWallViscosity)**b)

        singlePhaseHeatTransferCoefficient = nusseltNumber * coolantBulkThermalConductivity / getChannelHydraulicDiameter(position)
        
        # Find adjusted coolant side HTC from nucleate boiling
        kl = getCoolantThermalConductivity(pressure, bulkTemperature)
        cpl = getCoolantHeatCapacity(pressure, bulkTemperature)
        rhol = getCoolantDensity(pressure, bulkTemperature)
        rhog = PropsSI("D", "T", bulkTemperature, "Q", 1, fuel)
        mul = getCoolantViscosity(pressure, bulkTemperature)
        mug = PropsSI("V", "T", bulkTemperature, "Q", 1, fuel)
        g = 9.81
        dT = abs(wallTemperature - coolantSaturationTemperature)
        dP = abs(coolantVaporPres - pressure)
        vaporEnthalpy = PropsSI("H", "T", bulkTemperature, "Q", 1, fuel)
        liquidEnthalpy = PropsSI("H", "T", bulkTemperature, "Q", 0, fuel)
        hvap = vaporEnthalpy - liquidEnthalpy
        sigma = PropsSI("I", "T", bulkTemperature, "Q", 0, fuel)
        hnb = 0.00122 * (kl**0.79)*(cpl**0.45)*(rhol**0.49)*(g**0.25)*(dT**0.24)*(dP**0.75) / ((sigma**0.5)*(hvap**0.24)*(rhog**0.24)*(mul**0.45))
        x = -cpl * (coolantSaturationTemperature - bulkTemperature) / hvap
        xtt = (((1-abs(x))/abs(x))**0.9)*((rhog/rhol)**0.5)*((mul/mug)**0.1)
        if((1/xtt)<=0.1):
            F = 1
        else:
            F = 2.35*((1/xtt)+0.213)**0.736
        S = 1/(1+2.53e-6*(reynoldsNumber*F**1.25)**1.17)
        nbCoolantHeatTransferCoefficient = F * singlePhaseHeatTransferCoefficient + S * hnb * ((wallTemperature - coolantSaturationTemperature)/(wallTemperature - bulkTemperature))

        # Find adjustment to coolant side HTC due to 'fin' effects of channel lands
        # 
        finThickness = (np.pi * 2 * getEngineRadius(position) - numberCoolingChannels * channelWidth) / numberCoolingChannels
        finEfficiency = np.tanh((getChannelDepth(position)/finThickness) * np.sqrt((2 * nbCoolantHeatTransferCoefficient * finThickness)/(innerJacketThermalConductivity))) / np.sqrt((2 * nbCoolantHeatTransferCoefficient * finThickness)/(innerJacketThermalConductivity))
        finHeatTransferCoefficient = nbCoolantHeatTransferCoefficient * ((channelWidth + finEfficiency * 2 * getChannelDepth(position))/(channelWidth + finThickness))
        
        return 1 * finHeatTransferCoefficient

    # finds the gas side wall temperature, coolant side wall temperature, coolant bulk temperature, steady state heat flux, and coolant pressure 
    # of a coolant channel segment (increment of z axis length based on numberChannelSegments variable)
    def getCoolantSegmentProperties(position, nextPosition, upstreamCoolantTemperature, upstreamCoolantPressure):
        gasRecoveryTemperature = getRecoveryTemperature(position)
        coolantVaporPressure = PropsSI("P", "T", upstreamCoolantTemperature, "Q", 0, fuel)
        coolantSaturationTemperature = PropsSI("T", "P", upstreamCoolantPressure, "Q", 0, fuel)
        coolantVelocity = coolantChannelFlowRate / (getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * getChannelCrossSectionalArea(position)) # [m/s]
        coolantReynoldsNumber = getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * coolantVelocity * getChannelHydraulicDiameter(position) / getCoolantViscosity(upstreamCoolantPressure, upstreamCoolantTemperature)
        gasSideWallTemp = minimizeHeatFluxDifference(position, nextPosition, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure) # [K]
        gasHeatTransferCoefficient = getGasSideHeatTransferCoefficient(position, gasSideWallTemp)
        heatFlux = gasHeatTransferCoefficient * (gasRecoveryTemperature - gasSideWallTemp)
        coolantSideWallTemp = -(heatFlux * innerJacketWallThickness / innerJacketThermalConductivity - gasSideWallTemp)
        coolantTemperature = upstreamCoolantTemperature + (1 / (coolantChannelFlowRate * getCoolantHeatCapacity(upstreamCoolantPressure, upstreamCoolantTemperature))) * (heatFlux * getChannelSegmentLength(position, nextPosition) * getChannelLandCenteredWidth(position)) # [K]
        coolantHeatTransferCoefficient = getCoolantHeatTransferCoefficient(position, nextPosition, coolantVelocity, upstreamCoolantTemperature, coolantSideWallTemp, upstreamCoolantPressure)
        wallHeatFlux = innerJacketThermalConductivity * (gasSideWallTemp - coolantSideWallTemp) / innerJacketWallThickness
        coolantHeatFlux = coolantHeatTransferCoefficient * (coolantSideWallTemp - upstreamCoolantTemperature)
        channelFrictionFactor = getFrictionFactor(coolantReynoldsNumber, channelAbsoluteRoughness, getChannelHydraulicDiameter(position))
        #segmentPressureDrop = channelFrictionFactor * (getChannelSegmentLength(position, nextPosition)/getChannelHydraulicDiameter(position)) * 2 * getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) * (coolantVelocity ** 2) # [Pa]
        segmentPressureDrop = channelFrictionFactor * getChannelSegmentLength(position, nextPosition) / getChannelHydraulicDiameter(position) * getCoolantDensity(upstreamCoolantPressure, upstreamCoolantTemperature) / 2 * (coolantVelocity ** 2)
        coolantPressure = upstreamCoolantPressure - segmentPressureDrop
        maxJacketStress = ((coolantPressure-getGasPressure(position))/2) * ((channelWidth/innerJacketWallThickness)**2) + (innerJacketElasticModulus*innerJacketThermalExpansion*heatFlux*innerJacketWallThickness)/(2*(1-innerJacketPoissonRatio)*innerJacketThermalConductivity)

        # Determine if solution is below critical heat flux
        coolantSaturationTemperature = PropsSI("T", "P", upstreamCoolantPressure, "Q", 0, fuel)
        criticalHeatFlux = 1634246.178 * (0.1003 + 0.05264 * np.sqrt((coolantVelocity * 3.2808) * (1.8 * abs(coolantSaturationTemperature - upstreamCoolantTemperature)))) # [W/m^2]
        criticalHeatFluxMargin = round(1-(heatFlux/criticalHeatFlux), 2)*100 # percent
        if (criticalHeatFluxMargin < 15): print("CRITICAL HEAT FLUX SURPASSED: CHF Margin - " + str(criticalHeatFluxMargin) + "% " + "at z=" + str(position) + "m")
        
        return [gasSideWallTemp, coolantSideWallTemp, coolantTemperature, heatFlux, coolantPressure, coolantVaporPressure, maxJacketStress, gasHeatTransferCoefficient, coolantHeatTransferCoefficient]

    # calcualte gas side and coolant side heat transfer coefficients and find difference between the two heat fluxes for a set of parameters
    # steady state requires the gas side and coolant size heat fluxes to be equal, so this function must be iterated until it returns zero
    def heatFluxDifference(gasSideWallTemp, position, nextPosition, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure):
        
        gsHTC = getGasSideHeatTransferCoefficient(position, gasSideWallTemp) # [W/m2/K]
        gasHeatFlux = gsHTC * (getRecoveryTemperature(position) - gasSideWallTemp) # [W/m2]
        coolantSideWallTemperature = -(gasHeatFlux * innerJacketWallThickness / innerJacketThermalConductivity - gasSideWallTemp) # [K]
        csHTC = getCoolantHeatTransferCoefficient(position, nextPosition, coolantVelocity, upstreamCoolantTemperature, coolantSideWallTemperature, upstreamCoolantPressure) # [W/m2/K]
        coolantHeatFlux = csHTC * (coolantSideWallTemperature - upstreamCoolantTemperature)
        return abs(coolantHeatFlux - gasHeatFlux)

    # iterate gas side and coolant side heat fluxes until they equalize (steady state)
    # minimizes the heatFluxDifference function for the gas side wall temperature that equalizes the heat fluxes
    def minimizeHeatFluxDifference(position, nextPosition, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure):
        gasSideWallTemp = scipy.optimize.minimize_scalar(heatFluxDifference, bounds=(upstreamCoolantTemperature, getRecoveryTemperature(position)), method='bounded', args=(position, nextPosition, coolantVelocity, upstreamCoolantTemperature, upstreamCoolantPressure))['x'] # could experiment with tolerance here
        return gasSideWallTemp

    # Step through channels starting from nozzle exit
    # Collect heat flux, temperatures, pressures, etc from each segment
    # Store properties in vectors to plot against axial position along engine
    numberChannelSegments = 200
    axialSegmentLength = engineLength / numberChannelSegments # [m]
    axialPositions = np.zeros(numberChannelSegments) # [m]
    for i in range(len(axialPositions)-1):
        axialPositions[i+1] = axialPositions[i] + axialSegmentLength
    gasSideWallTemp = np.zeros(len(axialPositions)) # [K]
    coolantSideWallTemp = np.zeros(len(axialPositions)) # [K]
    coolantTemperature = np.zeros(len(axialPositions)) # [K]
    heatFlux = np.zeros(len(axialPositions)) # [W/m2/K]
    coolantPressure = np.zeros(len(axialPositions)) # [Pa]
    maxJacketStress = np.zeros(len(axialPositions)) # [Pa]
    nucleateBoilingCoolantHeatFlux = np.zeros(len(axialPositions)) # [W/m^2]
    gasHeatTransferCoefficient = np.zeros(len(axialPositions)) # [W/m^2/K]
    coolantHeatTransferCoefficient = np.zeros(len(axialPositions)) # [W/m^2/K]
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
        maxJacketStress[i] = props[6] # [Pa]
        gasHeatTransferCoefficient[i] = props[7] # [W/m^2/K]
        coolantHeatTransferCoefficient[i] = props[8] # [W/m^2/K]

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
    gasStaticPressure = np.zeros(len(zpoints))
    for i in range(len(gasStaticPressure)):
        gasStaticPressure[i] = getGasPressure(zpoints[i])
    channelDepth = np.zeros(len(zpoints))
    for i in range(len(channelDepth)):
        channelDepth[i] = getChannelDepth(zpoints[i])

    return [zpoints, engineRadius, machNumber, recoveryTemperature, gasStaticPressure, gasSideWallTemp, coolantSideWallTemp, coolantTemperature, heatFlux, coolantPressure, maxJacketStress, nucleateBoilingCoolantHeatFlux, channelDepth, gasHeatTransferCoefficient, coolantHeatTransferCoefficient]

##########################################################################################################################################################################################################################################################
            # STRUCTURAL GEOMETRY #                                                                                                                                                                                                                                sup fucker
##########################################################################################################################################################################################################################################################

# Engine Geometry
chamberLength = 0.08289 # [m]
chamberRadius = 0.02404 # [m]
lengthToThroat = 0.11026 # [m]
throatRadius = 0.0085 # [m]
throatRadiusOfCurvature = 0.005 # [m] what is a good value for this?
engineLength = 0.05707+lengthToThroat # [m]
nozzleExitRadius = 0.02945 # [m]

# Inner Jacket
innerJacketThermalConductivity = 22.5 # [W/m/K]
innerJacketWallThickness = 0.001 # [m]
innerJacketElasticModulus = 1.4e11 # [Pa]
innerJacketThermalExpansion = 2.0e-5
innerJacketPoissonRatio = 0.275

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
gasMolecularWeight = (28.01 * 0.236) + (44.01 * 0.087) + (2.0157 * 0.015) + (18.02 * 0.232) + (28.02 * 0.425) # get better chemical comp. from Lachlan CEA

##########################################################################################################################################################################################################################################################
            # COOLANT DEFINITION #                                                                                                                                                                                                                                sup fucker
##########################################################################################################################################################################################################################################################

fuel = 'Ethanol'
coolantWaterMassContent = 0.0
totalCoolantFlowRate = 0.0718 # [kg/s]
numberCoolingChannels = 36
initialCoolantPressure = 155 / pascalToPsi # [Pa]
initialCoolantTemperature = 280 # [K] could vary depending on orbit thermals, but ethanol pretty constant properties for wide temp & pres ranges

##########################################################################################################################################################################################################################################################
            # CHANNEL GEOMETRY #                                                                                                                                                                                                                                sup fucker
##########################################################################################################################################################################################################################################################

channelWidth = 0.000508 # [m]

#channelDepth = 0.0005 # [m] what is possible with our machining capabilities? what is optimal from an analysis side of things? seems like high aspect ratio (depth>width) is best

channelDepthChamber = 0.0015 # [m]
channelDepthThroat = 0.0005 # [m]
channelDepthNozzle = 0.0015 # [m]

channelAbsoluteRoughness = 0.000045 # [m] carbon steel

############
# RUN CODE #
############

[zpoints, engineRadius, machNumber, recoveryTemperature, gasStaticPressure, gasSideWallTemp, coolantSideWallTemp, coolantTemperature, heatFlux, coolantPressure, maxJacketStress, nucleateBoilingCoolantHeatFlux, channelDepth, gasHeatTransferCoefficient, coolantHeatTransferCoefficient] = runCoolingAnalysis(chamberLength, chamberRadius, lengthToThroat, throatRadius, throatRadiusOfCurvature, engineLength, nozzleExitRadius, innerJacketThermalConductivity, innerJacketWallThickness, gasSpecificHeatRatio, gasPrandtlNumber, gasViscosity, gasHeatCapacity, gasTemperature, chamberPressure, cstar, gasMolecularWeight, fuel, coolantWaterMassContent, totalCoolantFlowRate, numberCoolingChannels, initialCoolantPressure, initialCoolantTemperature, channelWidth, channelDepthChamber, channelDepthThroat, channelDepthNozzle, channelAbsoluteRoughness, innerJacketElasticModulus, innerJacketThermalExpansion, innerJacketPoissonRatio)

#fig, ax1 = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5), layout="constrained")

#ax1.set_xlabel("Axial (z) Position (from nozzle exit) [m]")
#ax1.set_ylabel("Temperature (K)")
#ax1.plot(zpoints, gasSideWallTemp, label="Gas Side Wall Temp [K]")
#ax1.plot(zpoints, coolantSideWallTemp, label="Coolant Side Wall Temp [K]")
#ax1.plot(zpoints, coolantTemperature, label="Coolant Temp [K]")
#ax1.tick_params(axis='y')

#ax2 = ax1.twinx()

#ax2.set_ylabel("Radius [m]")
#ax2.plot(zpoints, engineRadius, label="Engine Radius [m]")
#ax2.tick_params(axis='y')

#fig.tight_layout()
#plt.grid()
#plt.show()



def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

p1, = host.plot(zpoints, gasSideWallTemp, label="Gas Side Wall Temp [K]")
host.plot(zpoints, coolantSideWallTemp, label="Coolant Side Wall Temp [K]")
host.plot(zpoints, coolantTemperature, label="Coolant Temp [K]")
p2, = par1.plot(zpoints, engineRadius, label="Engine Radius [m]")
p3, = par2.plot(zpoints, heatFlux, label="Heat Flux [W/m^2]", color="tab:red")

host.set_xlim(0, 0.175)
host.set_ylim(0, 1000)
par1.set_ylim(0, 0.5)
par2.set_ylim(0, 5500000)

host.set_xlabel("Axial (z) Position (from nozzle exit) [m]")
host.set_ylabel("Temperature [K]")
par1.set_ylabel("Engine Radius [m]")
par2.set_ylabel("Heat Flux [W/m^2]")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines])

hf = plt.figure(2)
plt.plot(zpoints, heatFlux, label="Heat Flux [W/m^2]")
plt.plot(zpoints, nucleateBoilingCoolantHeatFlux, label="Nucleate Boiling Heat Flux [W/m^2]")
plt.legend

cd = plt.figure(3)
plt.plot(zpoints, channelDepth, label="Channel Depth [m]")
plt.legend

htc = plt.figure(4)
plt.plot(zpoints, gasHeatTransferCoefficient, label="Heat Transfer Coefficient [W/m^2/K]")
plt.plot(zpoints, coolantHeatTransferCoefficient, label="Nucleate Boiling Heat Flux [W/m^2/K]")

plt.show()


#plt.plot(zpoints, gasSideWallTemp, label="Gas Side Wall Temp [K]")
#plt.plot(zpoints, coolantSideWallTemp, label="Coolant Side Wall Temp [K]")
#plt.plot(zpoints, coolantTemperature, label="Coolant Temp [K]")
#plt.plot(zpoints, maxJacketStress, label="Max Inner Jacket Stress [Pa]")
#plt.plot(zpoints, heatFlux, label="Heat Flux [W/m^2]")
#plt.grid()
#plt.legend()
#plt.gca().set_xlabel("Axial (z) Position (from nozzle exit) [m]")
#plt.gca().set_ylabel("Temperature [K]")
#plt.gca().set_xlim([0, 0.5])
#plt.gca().set_ylim([0, 0.3])
#plt.show

print("Max Gas Side Wall Temp: " + str(max(gasSideWallTemp)) + " K")
#print("Max Inner Jacket Stress: " + str(max(maxJacketStress) / 1000000) + " MPa")
print("Max Coolant Temp: " + str(max(coolantTemperature)) + " K")
print("Min Coolant Pressure: " + str(np.min(coolantPressure[np.nonzero(coolantPressure)] * pascalToPsi)) + " psi")
print("Max Heat Flux: " + str(max(heatFlux) / 1000) + " kW/m^2")

    # TODO:
    # At each segment, find optimal channel 

    # debug fin effects
    # add better code to make pretty plots
    # comment all functions
    # add code evaluating stresses