import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate

# constants
kB = 1.38e-23
e = 1.6021e-19

#%%

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

def solveGroundedCapacitor(t_values, V_values, C, transistor1params, transistor2params, T):
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

        print(C*((Va_value - Va_values[-1])/(t-t_values[index-1]))/(1/(1/(R_ion_a) + 1/(R_ion_b))))

        return Va_value + C*((Va_value - Va_values[-1])/(t-t_values[index-1]))/(1/(1/(R_ion_a) + 1/(R_ion_b))) - Va(t, J1_value, J2_value, q1_value, q2_value)

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
t_values = sp.linspace(0, 60, 200)
V_values = sp.concatenate([sp.zeros(30), sp.ones(170)])

Va_values, J_values = solveGroundedCapacitor(t_values, V_values, 1, [1e8, 1e-8, 1.13e-11, 1.62e-10], [1e8, 1e-8, 1, 1], 300)
plt.plot(t_values, Va_values)
plt.title("VA vs time")
plt.show()
plt.plot(t_values, J_values)
plt.title("J vs time")
plt.show()

# %%
