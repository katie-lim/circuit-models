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

def solveCoupledCapacitor(t_values, V_values, Ru, Rd, C, transistor1params, transistor2params, T):
    R_ion_a, Ca, Js1a, Js2a = transistor1params
    R_ion_b, Cb, Js1b, Js2b = transistor2params

    V = interpolate.interp1d(t_values, V_values)

    Vu_values = [0]
    Vd_values = [0]
    integrand_values_u = [0]
    integrand_values_d = [0]
    Qu_values = [0]
    Qd_values = [0]

    def solveTwoTransistorStuff(truncated_t, new_Va, R, C, Js1, Js2, i, whichTransistor):
        voltage = interpolate.interp1d(truncated_t, new_Va)

        def integrand(t):
            return voltage(t) * sp.exp(2*t/(R*C))

        def q(i):
            if whichTransistor == "u":
                integrand_value = integrand_values_u[-1] + quad(integrand, t_values[i-1], t_values[i])[0]
            else:
                integrand_value = integrand_values_d[-1] + quad(integrand, t_values[i-1], t_values[i])[0]

            return sp.exp(-2*t_values[i]/(R*C))/R * integrand_value

        def V1(Va_value, q):
            return Va_value - q / C

        def V2(q):
            return q/C

        def Vn(q, V1, V2):
            return kB*T/e * sp.log( (sp.exp(2*e*V2/(kB*T)))*((Js2/Js1) + sp.exp(e*(V1-V2)/(kB*T))) / ((Js2/Js1) * sp.exp(2*e*(V2)/(kB*T))+1) )

        def transJ2(V2, Vn):
            return Js2*(sp.exp(e*(V2)/(kB*T))-sp.exp(e*(V2-Vn)/(kB*T)))

        q_value = q(i)
        V1_value = V1(new_Va[-1], q_value)
        V2_value = V2(q_value)
        Vn_value = Vn(q_value, V1_value, V2_value)
        J2_value = transJ2(V2_value, Vn_value)

        return J2_value, q_value

    def f(x, t):
        Vu_value, Vd_value = x
        index = sp.where(t_values == t)[0][0]

        truncated_t = t_values[0:len(Vu_values) + 1]
        new_Vu = Vu_values + [Vu_value]
        new_Vd = Vd_values + [Vd_value]

        Ju_value, Qu_value = solveTwoTransistorStuff(truncated_t, new_Vu, R_ion_a, Ca, Js1a, Js2a, index, "u")
        Jd_value, Qd_value = solveTwoTransistorStuff(truncated_t, new_Vd, R_ion_b, Cb, Js1b, Js2b, index, "u")

        dQudt = (Qu_value - Qu_values[-1])/(t-t_values[index-1])
        dQddt = (Qd_value - Qd_values[-1])/(t-t_values[index-1])

        dVuddt = ((Vu_value - Vu_values[-1]) - (Vd_value - Vd_values[-1]))/(t-t_values[index-1])

        eqn1 = (V_values[index] - Vu_value)/Ru + (V_values[index] - Vd_value)/Rd - dQudt - Ju_value - dQddt - Jd_value
        eqn2 = (1+Ru/Rd)*C*dVuddt - dQddt - Jd_value - Vd_value/Rd + (Ru/Rd)*(dQudt + Ju_value)

        return [eqn1, eqn2]


    for i in range(1, len(t_values)):
        solution = sp.optimize.fsolve(f, [Vu_values[-1], Vd_values[-1]], t_values[i])
        Vu_values.append(solution[0])
        Vd_values.append(solution[1])

        def integrandU(t, R, C):
            #(voltage(t) * sp.exp(2*t/(R*C))
            return interpolate.interp1d(t_values[0:i+1], Vu_values[0:i+1])(t) * sp.exp(2*t/(R*C))

        def integrandD(t, R, C):
            #(voltage(t) * sp.exp(2*t/(R*C))
            return interpolate.interp1d(t_values[0:i+1], Vd_values[0:i+1])(t) * sp.exp(2*t/(R*C))

        integrand_values_u.append(integrand_values_u[-1] + quad(integrandU, t_values[i-1], t_values[i], args=(R_ion_a, Ca))[0])
        integrand_values_d.append(integrand_values_d[-1] + quad(integrandD, t_values[i-1], t_values[i], args=(R_ion_b, Cb))[0])

    Ic = [0]
    for i in range(1, len(Vu_values)):
        Ic.append( ((Vu_values[i] - Vd_values[i]) - (Vu_values[i-1] - Vd_values[i-1])) / (t_values[i] - t_values[i-1]))

    return Vu_values, Vd_values, Ic

#%%
t_values = sp.linspace(0, 30, 1000)
V_values = sp.concatenate([sp.zeros(50), sp.ones(950)])

Vu_values, Vd_values, Ic = solveCoupledCapacitor(t_values, V_values, 5, 5, 1e-2, [1e9, 1e-8, 1e-7, 1e-5], [1e9, 1e-8, 1e-8, 1e-11], 300)
plt.plot(t_values, Vu_values)
plt.title("Vu vs time")
plt.show()
plt.plot(t_values, Vd_values)
plt.title("Vd vs time")
plt.show()
plt.plot(t_values, Ic)
plt.title("Ic vs time")
plt.show()

# %%
