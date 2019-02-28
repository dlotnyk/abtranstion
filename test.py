# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:17:36 2019

@author: dlotnyk
"""
# include needed libraries
import numpy as np
import matplotlib.pyplot as plt

# import test.txt file
 
path = "c:\\dima\\proj\\test.dat" # notice that \\ is used
data = np.genfromtxt(path, unpack=True, skip_header=1) # skip head import all columns
data1 = np.genfromtxt(path, unpack=True, skip_header=1, usecols = (2, 0)) 
# plot using matplotlib
# plot on the separate figures
fig1 = plt.figure(1, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel(r'$time^2$')
ax1.set_xlabel('time')
ax1.set_title('for ylabel I used latex code starts with r')
ax1.plot(data[0], data[1], color='green',lw=1) # plot with lines
plt.grid()

fig2= plt.figure(2, clear = True)
ax2 = fig2.add_subplot(111)
ax2.set_ylabel(r'$\sqrt{time}$')
ax2.set_xlabel('time')
ax2.set_title('square roots')
ax2.scatter(data1[0], data1[1], color='green',s=5) # plot with points
plt.grid()

# plot at 1 figure
fig3 = plt.figure(3, clear = True)
ax3 = fig3.add_subplot(211) # difference is in here
ax3.set_ylabel(r'$time^2$')
ax3.set_xlabel('time')
ax3.set_title('top plot 211')
ax3.plot(data[0], data[1], color='green',lw=1) # plot with lines
plt.grid()
ax4 = fig3.add_subplot(212)
ax4.set_ylabel(r'$\sqrt{time}$')
ax4.set_xlabel('time')
ax4.set_title('bottom plot 212')
ax4.scatter(data[0], data[1], color='green',s=5) # plot with points
plt.grid()

plt.show() # important to use only one plt.show() command right at the end