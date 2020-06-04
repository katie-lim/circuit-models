#%%
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
# constants
kB = 1.38e-23
e = 1.6021e-19

def solveTransistor(t_values, V_values, R, C, Js, T):
    voltage = interpolate.interp1d(t_values, V_values)

    def integrand(t):
        return voltage(t) * sp.exp(2*t/(R*C))

    #We need to firstly solve the top section of the circuit. All that needs to be done is computing the RHS integral.

    def q(t):
        return sp.exp(-2*t/(R*C))/R * quad(integrand, 0, t)[0]

    q = sp.vectorize(q) # Allow q(t) to take arrays as input

    #Solve the ODE
    q_values = q(t_values)

    #Now we must consider the transistor
    def V1(t):
        return voltage(t) - q(t) / C

    def transJ(t):
        return Js * (sp.exp(e*V1(t)/(kB*T)) - sp.exp(e*(V1(t)-voltage(t))/(kB*T)))

    V1_values = V1(t_values)
    J_values = transJ(t_values)

    return q_values, V1_values, J_values

def plotResults(t_values, V_values, q_values, V1_values, J_values):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(6, 8))

    ax1.plot(t_values, V_values)
    ax1.set_title("applied voltage vs time")

    ax2.plot(t_values, q_values)
    ax2.set_title("charge vs time")

    ax3.plot(t_values, J_values)
    ax3.set_title("current vs time")

    ax4.plot(t_values, V1_values)
    ax4.set_title("V1 vs time")

    plt.tight_layout()

    plt.show()
    return

def solveTwoTransistor(t_values, V_values, R, C, Js1, Js2, T):
    # First, find the solution for the single transistor
    q_values, V1_values, J_values = solveTransistor(t_values, V_values, R, C, Js1, T)

    # We must now solve for Vn(t), we use the fact that V2=q/c.
    def Vn(t):
        index = []
        for i in t:
            index.append(sp.where(t_values == i)[0][0])

        return kB*T/e*sp.log(((Js2/Js1) * sp.exp(2*e*q_values[index]/(C*kB*T)) + sp.exp(e*(q_values[index]/C + V1_values[index])/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*q_values[index]/(C*kB*T))+1))

    Vn_values = Vn(t_values)

    # Now that we have Vn, we will calculate the current transJ2
    def transJ2(t):
        index = []
        for i in t:
            index.append(sp.where(t_values == i)[0][0])

        return Js2*(sp.exp(e*q_values[index]/(C*kB*T))-sp.exp(e*(q_values[index]/C-Vn_values[index])/(kB*T)))

    J2_values = transJ2(t_values)

    return Vn_values, J2_values, q_values

#%%
t_values = sp.linspace(0, 30, 400)
V_values = sp.concatenate([sp.zeros(100), sp.ones(300)])

Vn_values, J_values, q_values = solveTwoTransistor(t_values, V_values, 1e6, 1e-6, 1e-10, 1e-8, 300)

plt.plot(t_values, V_values)
plt.title("applied voltage vs time")
plt.show()
plt.plot(t_values, Vn_values)
plt.title("Vn vs time")
plt.show()
plt.plot(t_values, J_values)
plt.title("current vs time")
plt.show()

#%%

def solveVa(t_values, V_values, R, R_ion, C, Js1, Js2, T):
    Va = [0]

    def f(Va_value, t):
        index = sp.where(t_values == t)[0][0]

        truncated_t = t_values[0:len(Va) + 1]
        new_Va = Va + [Va_value]

        Vn_values, J_values, q_values = solveTwoTransistor(truncated_t, new_Va, R_ion, C, Js1, Js2, T)

        return (Va_value - V_values[index] + R*(J_values[index] + Va_value/R_ion - 2*q_values[index]/(R_ion*C)))

    Va_0_guess = 0

    solution = sp.optimize.fsolve(f, Va_0_guess, t_values[1])[0]
    Va.append(solution)

    for i in range(2, len(t_values)):
        Va.append(sp.optimize.fsolve(f, Va[-1], t_values[i])[0])

    return Va

# %%
