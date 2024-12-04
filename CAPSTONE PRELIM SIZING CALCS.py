import numpy as np
import matplotlib.pyplot as plt
import csv
from pandas import DataFrame

#### Mission Parameters ####
# Constants and Input Params
mu = 3.986*10**5                                    # Earth Gravitational Constant [km3/s2]
r1 = 6908                                           # SpaceX rideshare reployment radius [km]
target_apogee = 9000                                # Desired final altitude [km]
g0 = 9.81 / 1000                                    # Acceleration due to gravity constant [km/s^2]
Isp_target = 280                                    # Assumed Specific Impulse (s)

# Initialize arrays of input values
a = np.linspace(r1,5*r1,1001)                       # Semi-Major Axes of raised orbit 
Isp = np.linspace(100, 300, 100)                    # Specific impulse 
MF = np.linspace(0.1, 3, 100)                       # Mass Fraction (values start from 0.1 to avoid log(0) issues)

# Compute final radius of the raised orbits
r2 = 2*a-r1                                         # Final (circular) radius after Hohmann Transfer

## Compute total Delta V required to raise orbit to array of final radii (with Hohmann Transfer)
# See Diagram on https://app.nuclino.com/SARP/CAPSTONE/Preliminary-Parameter-Sizing-3ac46d0b-8597-4bdd-a0e6-46a4a0a05a09s for visuals

# Delta V to raise the orbit (Burn 1)
dV_raise = (((2*mu/r1)-(mu/a))**.5)-((mu/r1)**.5)   

# Delta V to circularize the orbit (Burn 2)
dV_circ = (mu/r2)**.5-((2*mu/r2)-(mu/a))**.5

# Total Delta V to complete Hohmann Transfer
dV_tot = dV_raise + dV_circ

## Function determines Delta V required to reach target apogee
    # Inputs: array of final radii (r2), array of Delta V to reach final radii (dV)
    # Outputs: final radius @ target altitude, Delta V to reach target radius

def apogee(r2,dV):
    for i in range(len(r2)):
        if r2[i] >= target_apogee:
            apogee = r2[i]
            dV_nom = dV[i]
            return([apogee,dV_nom])

# Determine Delta V using apogee function
dV_nom = apogee(r2,dV_tot)[1]                       # Delta V required to reach target apogee

# Determines error in apogee function
r2_nom = apogee(r2,dV_tot)[0]                       
r2_error = (r2_nom-target_apogee)/target_apogee
print(f'Error in apogee = {r2_error*100:.4} %')

## Function determines Delta V from Tsiolkovsky Rocket Equation
    # Inputs: Array of Specific Impulses (Isp), and array of Mass Fractions (MF)
    # Outputs: Delta V for combination of Isp and MFs 

def compute_dV(Isp, MF):
    return Isp * g0 * np.log(MF)

# Create a 2D grid of values for Isp and Mass Fraction
X, Y = np.meshgrid(MF, Isp)

# Compute Delta V values for combinations on of Isp and Mass Fraction
Z = compute_dV(Y, X)  # Note: X is MF, Y is Isp

# Create 2D contour plot with Isp and Mass Fraction on the axes and delta V on the contour
plt.figure(figsize=(10, 7))
contour = plt.contourf(X, Y, Z, 20, cmap='plasma')
plt.colorbar(contour, label='Delta V (m/s)')

## Create a line on contour of Isp and MF combinations that deliver the Delta V required to reach target orbit
delta_v_target = dV_nom

# Find the coordinates where Delta V is approximately the target within 0.005 km/s
delta_v_error = 0.005
indices = np.where(np.abs(Z - delta_v_target) < delta_v_error)

# Extract the x, y coordinates for those indices
x_line = X[indices]
y_line = Y[indices]
z_line = Z[indices]

# Plot the line of Delta Vs on the contour plot
plt.plot(x_line, y_line, color='r', linewidth=2, label=f'Delta V = {delta_v_target:.3f}')

# Add a horizontal line at target Isp
plt.axhline(y=Isp_target, color='b', linestyle='--', linewidth=2, label=f'Expected Isp = {Isp_target:.1f}')

# Calculate the mass fraction (MF) at target Isp and required delta V
mf_target = np.exp(delta_v_target / (Isp_target * g0))

# Plot the mass fraction value with a vertical line
plt.axvline(x=mf_target, color='g', linestyle='--', linewidth=2, label=f'Required Mass Fraction = {mf_target:.3f}')

# Label the axes
plt.xlabel('Mass Fraction (MF)')
plt.ylabel('Specific Impulse (Isp)')
plt.title('2D Contour Plot of Delta V (Change in Velocity)')

# Show the legend
plt.legend()

# Display the plot
plt.show()

# Print the calculated mass fraction
print(f"Calculated mass fraction (MF) when Isp = {Isp_target} and Delta V = {dV_nom:.3f}: {mf_target:.3f}")

##### Part 2 Payload Mass ####

# Constants and Input Params
m_s = 50                                    # Assumed Structural Mass [kg]
MF = mf_target                              # Calculated mass fraction to achieve necessary delta V (assuming 280s of Isp)
rho_ox = 1.41                               # Density of oxidizer (90% HTP) [kg/L]
rho_f = 0.79                                # Density of fuel (100% Ethanol) [kg/L]
OF = 4.5                                    # Oxidizer:Fuel Ratio (from CEA)

# Target Design Params (Choose one or the other)
V_ox_target = 20                            # Target oxidizer volume (based on large SCUBA tank volumes) [L]
m_pay_target = 100                          # Target payload mass [kg]

# Input 0 below to solve using target Oxidizer Volume
# Input 1 below to solve using target Payload Mass
Solver_type = 1

# Initialize arrays of input values
m_pay = np.linspace(0, 200, 201)            # Payload mass [kg]

# Determine total propellant mass (Fuel + Oxidizer) from definition of Mass Fraction
m_p_tot = MF*(m_pay+m_s) - m_s - m_pay      # Total propellant mass [kg]

# Determine mass of fuel and oxidizer from mixture ratio
m_f = m_p_tot / (OF + 1)                    # Mass of fuel [kg]
m_ox = m_f * OF                             # Mass of oxidizer [kg]

# Determine Volume of fuel and oxidizer from definition of density
V_f = m_f / rho_f                           # Volume of fuel [L]
V_ox = m_ox / rho_ox                        # Volume of oxidizer [L]

# Initialize plot and plot Fuel and Ox Volume as functions of payload mass
fig, ax = plt.subplots()
ax.plot(m_pay, V_ox, color='blue', label='Oxidizer Volume')
ax.plot(m_pay, V_f, color='orange', label='Fuel Volume')

# Highlight large scuba tank volume with horizontal line
plt.hlines(V_ox_target, 0, 200, linestyle='--', label='Large SCUBA Tank')

# Definition of satellite size classifications
    # Source: https://www.nasa.gov/what-are-smallsats-and-cubesats/
nanosat_min = 1
nanosat_max = 10
microsat_min = 10
microsat_max = 100
minisat_min = 100
minisat_max = 180

# Plot sattelite size classifications on propellant volume vs payload mass figure
ax.fill_between(m_pay, V_ox, 0, where=(m_pay >= nanosat_min) & (m_pay <= nanosat_max), color='gray', alpha=0.5, label='Nanosat Range')
ax.fill_between(m_pay, V_ox, 0, where=(m_pay >= microsat_min) & (m_pay <= microsat_max), color='palegreen', alpha=0.5, label='Microsat Range')
ax.fill_between(m_pay, V_ox, 0, where=(m_pay >= minisat_min) & (m_pay <= minisat_max), color='plum', alpha=0.5, label='Minisat Range')

## Function calculates the payload mass for a given oxidizer volume
    # Inputs: Target Oxidizer Volume (currently based on large SCUBA tank)
    # Outputs: Maximum Payload mass to reach desired apogee (based on Mass Fraction)

def payload_for_oxidizer_volume(V_ox_target):
    m_pay = (V_ox_target * (OF + 1) * rho_ox) / ((MF - 1) * OF) - m_s
    return(m_pay)

def oxidizer_volume_for_payload_mass(m_pay_target):
    V_ox = (MF-1)*(m_pay_target + m_s)*OF/(OF+1)/rho_ox
    return(V_ox)

## If-else statement solves for oxidizer volume and payload mass depending on which solver mode selected
    # Inputs: Solver type, target oxidizer volume OR payload mass
    # Outputs: Ox volume and payload mass

if Solver_type == 0:
    print(f'Type 0 solver selected: Solving for payload mass given an oxidizer volume of {V_ox_target} L')
    # Calculate the corresponding maximum payload mass for target oxidizer volume
    m_pay_at_V_ox_target = payload_for_oxidizer_volume(V_ox_target)
    print(f"Payload mass corresponding to {V_ox_target} L oxidizer volume: {m_pay_at_V_ox_target:.2f} kg")
    V_ox = V_ox_target
    m_pay = m_pay_at_V_ox_target
    
elif Solver_type == 1:
    print(f'Type 1 solver selected: Solving for oxidizer volume required to deliver a {m_pay_target} kg payload to {target_apogee} km')
    # Calculate the required oxidizer volume to deliver target payload mass to apogee
    V_ox_at_m_pay_target = oxidizer_volume_for_payload_mass(m_pay_target)
    print(f'Oxidizer Volume corresponding to {m_pay_target} kg of payload: {V_ox_at_m_pay_target:.4} L')
    V_ox = V_ox_at_m_pay_target
    m_pay = m_pay_target

# Determine and print propellant masses and total spacecraft mass
m_ox = V_ox * rho_ox
m_f = V_ox/OF * rho_f
m_p = m_ox + m_f                                                 # Propellant mass with target volume of oxidizer
print(f'Total Propellant Mass = {m_p:.4} kg')

m0 = m_pay + m_p + m_s                                           # Total mass
print(f'Total Spacecraft Mass = {m0:.4} kg')


# Highlight the payload mass where V_ox = target ox volume 
ax.plot(m_pay, V_ox, 'ro')  # Red dot for the point
ax.text(m_pay + 5, V_ox + 0.2, f'm_pay: {m_pay} kg, V_ox: {V_ox:.3} L', color='red')

text_box = ax.text(0, 0, f"Params: Mass Fraction = {mf_target:.3f},  deltaV = {dV_nom:.3f} km/s, Isp = {Isp_target} s, Structural Mass = {m_s} kg", transform=ax.transAxes, 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

# Plot titles, legend, and axis labels
plt.legend()
plt.xlabel('Payload Mass (kg)')
plt.ylabel('Propellant Volume (L)')
plt.title("Oxidizer and Fuel Volume vs. Payload Mass")
plt.show()

# Bonus: Find total impulse to raise 
I = Isp_target*m0*g0
print(f'Total Impulse = {I:.5} N s ')

# Create Dataframe 
mission_param_outputs = [target_apogee,Isp_target,dV_nom]
labels = np.array(['Apogee [km]','Isp [s]','Delta V [m/s]'])
df = DataFrame({'Parameter':labels,'Value':mission_param_outputs})

df.to_excel('Master Parameter Sheet.xlsx', sheet_name = 'Mission Parameters', index=False)

mission_param_outputs = [target_apogee,Isp_target,dV_nom]
labels = np.array(['Apogee [km]','Isp [s]','Delta V [m/s]'])
df = DataFrame({'Parameter':labels,'Value':mission_param_outputs})

df.to_excel('Master Parameter Sheet.xlsx', sheet_name = 'Propellant Volume Sizing', index=False)