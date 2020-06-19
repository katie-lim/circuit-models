#%%
import scipy as sp
from scipy import signal
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
        return sp.exp(-2*t/(R*C))/R * quad(integrand, 0, t, epsrel=0.1)[0]

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

def solveTwoTransistorGeneral(t_values, V_values, VE_values, R, C, Js1, Js2, T):
    voltage = interpolate.interp1d(t_values, V_values)
    VE = interpolate.interp1d(t_values, VE_values)

    integrand_values = []
    q_values = []
    V1_values = []
    V2_values = []

    Vn_values = []
    J2_values = []

    def integrand(t):
        return (voltage(t) - VE(t)) * sp.exp(2*t/(R*C))

    def q(i):
        integrand_value = integrand_values[-1] + quad(integrand, t_values[i-1], t_values[i])[0]
        integrand_values.append(integrand_value)

        return sp.exp(-2*t_values[i]/(R*C))/R * integrand_values[-1]

    def V1(i):
        return V_values[i] - q_values[i] / C

    def V2(i):
        return q_values[i] / C + VE_values[i]

    def Vn(V1, V2, VE):
        return kB*T/e * sp.log( (sp.exp(2*e*V2/(kB*T)))*((Js2/Js1) * sp.exp(e*VE/(kB*T)) + sp.exp(e*(V1+VE-V2)/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*(V2-VE)/(kB*T))+1) )

    def transJ2(V2, VE, Vn):
        return Js2*(sp.exp(e*(V2-VE)/(kB*T))-sp.exp(e*(V2-Vn)/(kB*T)))

    for i in range(0, len(t_values)):
        if i == 0:
            integrand_values.append(0)
            q_values.append(0)
        else:
            q_values.append(q(i))

        V1_values.append(V1(i))
        V2_values.append(V2(i))
        Vn_values.append(Vn(V1_values[i], V2_values[i], VE_values[i]))
        J2_values.append(transJ2(V2_values[i], VE_values[i], Vn_values[i]))

    return Vn_values, J2_values, q_values

def solveTwoTransistor(t_values, V_values, R, C, Js1, Js2, T):
    voltage = interpolate.interp1d(t_values, V_values)

    integrand_values = []
    q_values = []
    V1_values = []

    Vn_values = []
    J2_values = []

    def integrand(t):
        return voltage(t) * sp.exp(2*t/(R*C))

    def q(i):
        integrand_value = integrand_values[-1] + quad(integrand, t_values[i-1], t_values[i])[0]
        integrand_values.append(integrand_value)

        return sp.exp(-2*t_values[i]/(R*C))/R * integrand_values[-1]

    def V1(i):
        return V_values[i] - q_values[i] / C

    def Vn(q, V1):
        return kB*T/e*sp.log(((Js2/Js1) * sp.exp(2*e*q/(C*kB*T)) + sp.exp(e*(q/C + V1)/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*q/(C*kB*T))+1))

    def transJ2(q, Vn):
        return Js2*(sp.exp(e*q/(C*kB*T))-sp.exp(e*(q/C-Vn)/(kB*T)))

    for i in range(0, len(t_values)):
        if i == 0:
            integrand_values.append(0)
            q_values.append(0)
        else:
            q_values.append(q(i))

        V1_values.append(V1(i))
        Vn_values.append(Vn(q_values[i], V1_values[i]))
        J2_values.append(transJ2(q_values[i], Vn_values[i]))

    return Vn_values, J2_values, q_values

#%%
#Resistor and transistor in series
def solveVa(t_values, V_values, R, R_ion, C, Js1, Js2, T):
    Va = [0]

    integrand_values = [0]

    def solveTwoTransistorStuff(truncated_t, new_Va, i):
        voltage = interpolate.interp1d(truncated_t, new_Va)

        def integrand(t):
            return voltage(t) * sp.exp(2*t/(R_ion*C))

        def q(i):
            integrand_value = integrand_values[-1] + quad(integrand, t_values[i-1], t_values[i])[0]

            return sp.exp(-2*t_values[i]/(R_ion*C))/R_ion * integrand_value

        def V1(Va_value, q):
            return Va_value - q / C

        def Vn(q, V1):
            return kB*T/e*sp.log(((Js2/Js1) * sp.exp(2*e*q/(C*kB*T)) + sp.exp(e*(q/C + V1)/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*q/(C*kB*T))+1))

        def transJ2(q, Vn):
            return Js2*(sp.exp(e*q/(C*kB*T))-sp.exp(e*(q/C-Vn)/(kB*T)))

        q_value = q(i)
        V1_value = V1(new_Va[-1], q_value)
        Vn_value = Vn(q_value, V1_value)
        J2_value = transJ2(q_value, Vn_value)

        return J2_value, q_value


    def f(Va_value, t):
        index = sp.where(t_values == t)[0][0]

        truncated_t = t_values[0:len(Va) + 1]
        new_Va = Va + [Va_value]

        J_value, q_value = solveTwoTransistorStuff(truncated_t, new_Va, index)
        return (Va_value - V_values[index] + R*(J_value + Va_value/R_ion - 2*q_value/(R_ion*C)))


    for i in range(1, len(t_values)):
        solution = sp.optimize.fsolve(f, Va[-1], t_values[i])[0]
        Va.append(solution)

        def integrand2(t):
            return interpolate.interp1d(t_values[0:i+1], Va[0:i+1])(t) * sp.exp(2*t/(R_ion*C))

        integrand_values.append(quad(integrand2, 0, t_values[i])[0])


    Vn_values, J_values, q_values = solveTwoTransistor(t_values, Va, R_ion, C, Js1, Js2, T)
    I = sp.array(J_values) + sp.array(Va)/R_ion - 2*sp.array(q_values)/(R_ion*C)

    return Va, J_values, I

#%%
# Two transistors in series
def solveMemristorsInSeries(t_values, V_values, transistor1params, transistor2params, T):
    R_ion_a, Ca, Js1a, Js2a = transistor1params
    R_ion_b, Cb, Js1b, Js2b = transistor2params

    V = interpolate.interp1d(t_values, V_values)

    Va_values = [0]

    def Va(t, J1, J2, Q1, Q2):
        return 1/(1/(R_ion_a) + 1/(R_ion_b)) * (J1 - J2 + 2*Q2/(R_ion_b*Cb) - 2*Q1/(R_ion_a*Ca) + V(t)/R_ion_a)

    integrand_values_a = [0]
    integrand_values_b = [0]

    def solveTwoTransistorStuff(truncated_t, new_Va, new_VE, R, C, Js1, Js2, i, whichTransistor):
        voltage = interpolate.interp1d(truncated_t, new_Va)
        VE = interpolate.interp1d(truncated_t, new_VE)

        def integrand(t):
            return (voltage(t) - VE(t)) * sp.exp(2*t/(R*C))

        def q(i):
            if whichTransistor == "a":
                integrand_value = integrand_values_a[-1] + quad(integrand, t_values[i-1], t_values[i])[0]
            else:
                integrand_value = integrand_values_b[-1] + quad(integrand, t_values[i-1], t_values[i])[0]

            return sp.exp(-2*t_values[i]/(R*C))/R * integrand_value

        def V1(Va_value, q):
            return Va_value - q / C

        def V2(q, VE):
            return q/C + VE

        def Vn(q, V1, V2, VE):
            return kB*T/e * sp.log( (sp.exp(2*e*V2/(kB*T)))*((Js2/Js1) * sp.exp(e*VE/(kB*T)) + sp.exp(e*(V1+VE-V2)/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*(V2-VE)/(kB*T))+1) )

        def transJ2(V2, VE, Vn):
            return Js2*(sp.exp(e*(V2-VE)/(kB*T))-sp.exp(e*(V2-Vn)/(kB*T)))

        q_value = q(i)
        V1_value = V1(new_Va[-1], q_value)
        V2_value = V2(q_value, new_VE[-1])
        Vn_value = Vn(q_value, V1_value, V2_value, new_VE[-1])
        J2_value = transJ2(V2_value, new_VE[-1], Vn_value)

        return J2_value, q_value

    def f(Va_value, t):
        index = sp.where(t_values == t)[0][0]

        truncated_t = t_values[0:len(Va_values) + 1]
        truncated_V = V_values[0:len(Va_values) + 1]
        new_Va = Va_values + [Va_value]

        J1_value, q1_value = solveTwoTransistorStuff(truncated_t, truncated_V, new_Va, R_ion_a, Ca, Js1a, Js2a, index, "a")
        J2_value, q2_value = solveTwoTransistorStuff(truncated_t, new_Va, sp.zeros(len(Va_values) + 1), R_ion_b, Cb, Js1b, Js2b, index, "b")

        return Va_value - Va(t, J1_value, J2_value, q1_value, q2_value)

    for i in range(1, len(t_values)):
        solution = sp.optimize.fsolve(f, Va_values[-1], t_values[i])[0]
        Va_values.append(solution)

        def integrandA(t, R, C):
            #(voltage(t) - VE(t)) * sp.exp(2*t/(R*C))
            return (interpolate.interp1d(t_values[0:i+1], V_values[0:i+1])(t) - interpolate.interp1d(t_values[0:i+1], Va_values[0:i+1])(t)) * sp.exp(2*t/(R*C))

        def integrandB(t, R, C):
            #(voltage(t) * sp.exp(2*t/(R*C))
            return interpolate.interp1d(t_values[0:i+1], Va_values[0:i+1])(t) * sp.exp(2*t/(R*C))

        integrand_values_a.append(integrand_values_a[-1] + quad(integrandA, t_values[i-1], t_values[i], args=(R_ion_a, Ca))[0])
        integrand_values_b.append(integrand_values_b[-1] + quad(integrandB, t_values[i-1], t_values[i], args=(R_ion_b, Cb))[0])

    Vn_values, J_values, q_values = solveTwoTransistor(t_values, Va_values, R_ion_b, Cb, Js1b, Js2b, T)
    #Vn_values, J_values, q_values = solveTwoTransistorGeneral(t_values, V_values, Va_values, R_ion_a, Ca, Js1a, Js2a, T)

    return Va_values, J_values
#%%
t_values = sp.linspace(0, 60, 100)
V_values = sp.concatenate([sp.zeros(15), sp.ones(85)])
#VE_values = sp.concatenate([sp.zeros(20), 0.1*sp.ones(80)])

#Vn_values, J_values, q_values = solveTwoTransistor(t_values, V_values, 5, 1, 1e-8, 1e-10, 300)
#Vn_values, J_values, q_values = solveTwoTransistorGeneral(t_values, V_values, VE_values, 5, 1, 1e-8, 1e-10, 300)

# plt.plot(t_values, V_values)
# plt.title("applied voltage vs time")
# plt.show()
# plt.plot(t_values, Vn_values)
# plt.title("Vn vs time")
# plt.show()
# plt.plot(t_values, J_values)
# plt.title("current vs time")
# plt.show()

# Resistor and memristor in series
# Va_values, J_values, I = solveVa(t_values, V_values, 100, 1e7, 1e-6, 1e-7, 1e-5, 300)
# plt.plot(t_values, Va_values)
# plt.title("VA vs time")
# plt.show()
# plt.plot(t_values, I)
# plt.title("I vs time")
# plt.show()

# Two memristors in series
Va_values, J_values = solveMemristorsInSeries(t_values, V_values, [1e9, 1e-8, 1e-4, 5.46e-14], [3e9, 1e-8, 1.00e-15, 1.13e-11], 300)
plt.plot(t_values, Va_values)
plt.title("VA vs time")
plt.show()
plt.plot(t_values, J_values)
plt.title("J vs time")
plt.show()

# %%
# do stuff
t_values = sp.linspace(0, 60, 100)
V_values = sp.concatenate([sp.zeros(15), sp.ones(85)])

# Js1_values = sp.logspace(-20, -2, 60)
# Js2_values = sp.logspace(-20, -2, 60)
Js1_values_a = sp.logspace(-16, -14, 6)
Js2_values_a = sp.logspace(-12, -10, 6)

#1.00e-15, 1.13e-11

pairs = [(a, b) for a in Js1_values_a for b in Js2_values_a]

for pair in pairs:
    Va_values, J_values = solveMemristorsInSeries(t_values, V_values, [1e9, 1e-8, 1.00e-4, 2.68e-14], [3e9, 1e-8, pair[0], pair[1]], 300)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 8))
    ax1.plot(t_values, Va_values)
    ax1.set_title("VA vs time")
    ax2.plot(t_values, J_values)
    ax2.set_title("J vs time")

    title = "Js1 = {:.2e}".format(pair[0]) + ", Js2 = {:.2e}".format(pair[1])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.90)
    plt.savefig("graphs3/"+title+".png")
    plt.show()


# %%

# def solveSeriesCircuit(t_values, V_values, Ra, Rb, C, T, transistor1params, transistor2params):
#     R_ion_a, Ca, Js1a, Js2a = transistor1params
#     R_ion_b, Cb, Js1b, Js2b = transistor2params

#     Va = [0]
#     Vb = [0]

#     def equations(p, t):
#         Va_value, Vb_value = p

#         truncated_t = t_values[0:len(Va) + 1]
#         new_Va = Va + [Va_value]
#         new_Vb = Vb + [Vb_value]

#         Va1, Ja, Ia2 = solveVa(truncated_t, new_Va, Ra, R_ion_a, Ca, Js1a, Js2a, T)
#         Vb2, Jb, Ib2 = solveVa(truncated_t, new_Vb, Rb, R_ion_b, Cb, Js1b, Js2b, T)

#         Ia2 = interpolate.interp1d(truncated_t, Ia2)
#         Ib2 = interpolate.interp1d(truncated_t, Ib2)

#         def integrand(t):
#             return sp.exp(-t/((Ra + Rb)*C))/(Ra+Rb) * (Rb*Ib2(t) - Ra*Ia2(t))

#         def f(Va_value, Vb_value, t):
#             index = sp.where(t_values == t)[0][0]

#             return Va_value*Rb + Vb_value*Ra + (Ia2(t)+Ib2(t)+Ja[index]+Jb[index])*Ra*Rb -V_values[index]*(Rb+Ra)

#         def g(Va_value, Vb_value, t):
#             index = sp.where(t_values == t)[0][0]

#             def Q(Va_value, Vb_value, t):
#                 return sp.exp(t/((Ra + Rb)*C)) * quad(integrand, 0, t, epsrel=0.1)[0]

#             return Vb_value + Q(Va_value, Vb_value, t) - Va_value


#         return (f(Va_value, Vb_value, t), g(Va_value, Vb_value, t))

#     Va_0_guess = 0
#     Vb_0_guess = 0

#     solution = sp.optimize.fsolve(equations, (Va_0_guess, Vb_0_guess), t_values[1], xtol=0.1)
#     Va.append(solution[0])
#     Vb.append(solution[1])

#     for i in range(2, len(t_values)):
#         solution = sp.optimize.fsolve(equations, (Va[-1], Vb[-1]), t_values[i], xtol=0.1)
#         print(solution)
#         Va.append(solution[0])
#         Vb.append(solution[1])

#     return Va, Vb


# Va, Vb = solveSeriesCircuit(t_values, V_values, 1, 1, 1, 300, [1,1,1e-8,1e-10], [1,1,1e-10,1e-8])

# plt.plot(t_values, Va)
# plt.show()
# plt.plot(t_values, Vb)
# plt.show()