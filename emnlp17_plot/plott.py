import matplotlib.pyplot as plt
from pylab import *

hh = open("Hoffmann.txt", "r")
mi = open("Mintz.txt", "r")
ml = open("MIMLRE.txt", "r")
g = open("lykATT.txt", "r")
h = open("lykONE.txt", "r")
o = open("att+soft-label.txt", "r")
tt = open("one+soft-label.txt", "r")


xx = []
yy = []
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []

x4, y4 = [], []
x5, y5 = [], []
x6, y6 = [], []
for line in hh:
    line = line.strip()
    it = line.split()
    x4.append(float(it[0]))
    y4.append(float(it[1]))
for line in mi:
    line = line.strip()
    it = line.split()
    x5.append(float(it[0]))
    y5.append(float(it[1]))
for line in ml:
    line = line.strip()
    it = line.split()
    x6.append(float(it[0]))
    y6.append(float(it[1]))


for line in g:
    line = line.strip()
    it = line.split()
    xx.append(float(it[0]))
    yy.append(float(it[1]))

for line in h:
    line = line.strip()
    it = line.split()
    x1.append(float(it[0]))
    y1.append(float(it[1]))

for line in o:
    line = line.strip()
    it = line.split()
    x2.append(float(it[0]))
    y2.append(float(it[1]))

for line in tt:
    line = line.strip()
    it = line.split()
    x3.append(float(it[0]))
    y3.append(float(it[1]))
plt.ylim(0.3,1.0)
plt.xlim(0,0.4)
# plot(y, x, "r")
# plt.show()

plot(x5, y5, "-g", marker="o", markevery=160,label="Mintz")
plot(x4, y4, "-c", marker="2", markevery=160,label="MultiR")
plot(x6, y6, "-k", marker=",", markevery=160,label="MIMLRE")
plot(y1, x1, "-b", marker="d", markevery=100,label="PCNN-ONE")
plot(yy, xx, "y", marker="*", markevery=60, label="PCNN-ATT")
plot(y3, x3, "m", marker="x", markevery=60,label="PCNN-ONE+soft-label")
plot(y2, x2, "r", marker="+", markevery=60,label="PCNN-ATT+soft-label")

ylabel('Precision')
xlabel('Recall')
legend()
grid()
savefig('fig.eps', format='eps',dpi=2000, bbox_inches='tight')
show()

