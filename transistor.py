import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# constants
kB = 1.38e-23
e = 1.6021e-19

R = 1
C = 1

def voltage(t):
    return 0.5*(sp.sign(t/10 - 5) + sp.sign(1 - (t/10 - 5))) # a rectangle function
    # if t < 40:
    #     return 0
    # elif t < 50:
    #     return 1
    # else:
    #     return 0

def integrand(t):
    return voltage(t) * sp.exp(2*t/(R*C))

#We need to firstly solve the top section of the circuit. All that needs to be done is computing the RHS integral.

def q(t):
    return sp.exp(-2*t/(R*C))/R * quad(integrand,0,t)[0]

q = sp.vectorize(q) # Allow q(t) to take arrays as input

def V1(t):
    return voltage(t) - q(t)

def transJ(t):
    return Js * (sp.exp(e*V1(t)/(kB*T)) - sp.exp(e*(V1(t)-voltage(t))/(kB*T)))

#Solve the ODE
t = sp.linspace(0, 100, 200)
q_values = q(t)

#Now we must consider the transistor
Js = 1
T = 300
J_values = transJ(t)

#Finally, plot the results
def plotResults():
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 8))

    ax1.plot(t, voltage(t))
    ax1.set_title("applied voltage vs time")

    ax2.plot(t, q_values)
    ax2.set_title("charge vs time")

    ax3.plot(t, J_values)
    ax3.set_title("current vs time")

    plt.tight_layout()

    plt.show()
    return

plotResults()