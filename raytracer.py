import os, sys
import numpy as np

class raytracer:
	def __init__(self, c, dim):
		self.c = c
		self.dim = dim

	# numerical estimate of the gradient of c
	def grad_c(self, x, dx):
		k = np.zeros(np.size(x))
		grad_c = np.zeros(np.size(x))
		
		for i in range(0,self.dim):
			k[i] = dx
			grad_c[i] =  (-self.c(x+2.0*k) + 8.0*self.c(x+k) - 8.0*self.c(x-k) + self.c(x-2.0*k))/(12. * dx)
			k[i] = 0    
		return grad_c
   	
	# Hamiltonian resulting from wave-speed
	def HAMIL(self, x, xi):	
		return 0.5 * (self.c(x)**2 * np.inner(xi,xi) -1)
	
	# x-gradient of Hamiltonian
	def dHAMIL_dxi(self, x, xi):
		return self.c(x)**2 * xi
	
	# xi-gradient of Hamiltonian, numerically estimated
	def dHAMIL_dx(self, x, xi, dx):
		k = np.zeros(np.size(x))
		dHAMIL_dx = np.zeros(np.size(x))
	
		for i in range(0, self.dim):
			k[i] = dx
			dHAMIL_dx[i] = (-self.HAMIL(x+2.0*k,xi) + 8.0*self.HAMIL(x+k,xi) - 8.0*self.HAMIL(x-k,xi) + self.HAMIL(x-2.0*k,xi))/(12.0 * dx)		
			k[i] = 0
		return dHAMIL_dx
	
	# norm of covector in metric g = c^{-2}e, so:
	#        \|xi\|_g = c(x) \|xi\|_e
	def covec_norm_g(self, x, xi):
		return self.c(x) * np.sqrt(np.inner(xi,xi))
	
	# normalize a covector
	def g_normalize_covec(self, x, xi):
		return (1.0 / self.covec_norm_g(x,xi)) * xi
	
	
	# stopping condition
	def out_of_bounds(self, x):
		return x[1] < 0.0		
	
	# http://en.wikipedia.org/wiki/Eikonal_equation
	def solve_hamilton_jacobi(self, x0, xi0, n, T):
		# dx for taking gradients
		dx = .001
	
		travel_time   = 0.0	
		abs_max_hamil = 0.0	
		travel_time   = 0.0
		
		# the step-size and it's multiples
		h        =  T / float(n)
		half_h   =  0.5 * h
		h_over_6 =  h / 6.0
		
	 	# initialize
		x_cur  = x0[:]
		xi_cur = self.g_normalize_covec(x0, xi0)
		
		# first term in xs
		xs = np.zeros((n, self.dim))
	
		for i in range(n):
			xs[i,:] = x_cur[:]
			x_old  = x_cur[:]
			xi_old = xi_cur[:]
		
			# following line is for de-bugging
			if abs(self.HAMIL(x_old,xi_old)) > abs_max_hamil :
				abs_max_hamil = abs(self.HAMIL(x_old,xi_old))
				
			travel_time = travel_time + h
			
			kx_1  =  self.dHAMIL_dxi(x_old, xi_old)
			kxi_1 = -self.dHAMIL_dx (x_old, xi_old, dx)
			
			kx_2  =  self.dHAMIL_dxi(x_old + half_h * kx_1, xi_old + half_h * kxi_1)
			kxi_2 = -self.dHAMIL_dx (x_old + half_h * kx_1, xi_old + half_h * kxi_1, dx)
			
			kx_3  =  self.dHAMIL_dxi(x_old + half_h * kx_2, xi_old + half_h * kxi_2)
			kxi_3 = -self.dHAMIL_dx (x_old + half_h * kx_2, xi_old + half_h * kxi_2, dx)
			
			kx_4  =  self.dHAMIL_dxi(x_old + h * kx_3, xi_old + h * kxi_3)
			kxi_4 = -self.dHAMIL_dx (x_old + h * kx_3, xi_old + h * kxi_3, dx)
			
			x_cur  =  x_old + h_over_6 * (kx_1  + 2*kx_2  + 2* kx_3  + kx_4 ) 
			xi_cur = xi_old + h_over_6 * (kxi_1 + 2*kxi_2 + 2* kxi_3 + kxi_4)
	
			if self.out_of_bounds(x_cur):			
				break
	
		if self.out_of_bounds(x_cur):
			# estimate where the ray pierces the line y = 0 by solving for
			# tau* where:
			#            0 = y_i-1 + tau* (y_i - y_i-1)
			# then the total travel time and x-position should be estimated
			# by:
			#            t* = t_i-1 + tau* (t_i - t_i-1)
			#            x* = x_i-1 + tau* (x_i - x_i-1)
			tau_star = x_old[1] / (x_old[1] - x_cur[1])
			total_travel_time = h * i + tau_star * h
			x_final           = x_old + tau_star * (x_cur - x_old)

			for k in range(i+1,n):
				xs[k,:] = x_final

		else:
			total_travel_time = T

		return xs, total_travel_time
		
	def fire_ray_fan(self, x0, nthetas, n, T, theta_lo = 0.0, theta_hi = np.pi):
		data = []
		for i in range(nthetas + 1):
			theta_i = theta_lo + (i * (theta_hi - theta_lo)) / nthetas		
			xi0 =  np.array([np.cos(theta_i), np.sin(theta_i)])
			xs, total_travel_time  =  self.solve_hamilton_jacobi(x0, xi0, n, T)
			data.append({"xs" : xs[:], 'total_travel_time' : total_travel_time})
		return data
