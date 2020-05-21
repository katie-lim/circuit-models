import scipy as sp
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

# constants
k_B = 1.38e-23
q = 1.6021e-19

R = 1
C = 1

def voltage(t):
    if t < 50:
        return 0
    if t < 60:
        return 10
    else:
        return 0

def integrand(t):
    return voltage(t) * math.exp(2*t/(R*C))

#We need to firstly solve the top section of the circuit. All that needs to be done is computing the RHS integral.
def V_1(t):
    return voltage(t) - math.exp(-2*t/(R*C))/(R*C) * quad(integrand,0,t)[0]

#Plotting V_1
t = sp.arange(0, 101, 1)
V = [V_1(i) for i in range(0,101)]
plt.plot(t, V)
plt.show()

#Now we must consider the transistor
Js=1
T=300

def transJ(t):
    return Js*(math.exp(q*V_1(t)/(k_B*T))-math.exp(q*(V_1(t)-voltage(t))/(k_B*T)))

#Plotting transJ
J = [transJ(i) for i in range(0,101)]
plt.plot(t, J)
plt.show()
