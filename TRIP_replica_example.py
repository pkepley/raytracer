import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from raytracer import raytracer

def c(x):
	#return (1.0+0.5*x[1]-0.75*np.exp(-4.0*((x[0]*x[0])+(x[1]-0.75)*(x[1]-0.75))))
	#return (1.0+0.15*np.cos( 0.5*np.pi*x[0] + 1.5*np.pi*x[1]))
	#return (1.0 + x[1])
	#return 1.0+0.25*np.exp(-4.0*((x[1]-0.5)*(x[1]-0.5)))
	#return 1.0+0.5*np.exp(-64.0*((x[1]-0.5)*(x[1]-0.5)))
	#return 1.0 - 0.5* np.exp(-16.0*((x[0]*x[0]) + (x[1]-0.5)*(x[1]-0.5)));
	return 1.0 + 0.5*x[1] - 0.5*np.exp(-4.0*x[0]*x[0] - 4.0*(x[1]-0.375)*(x[1]-0.375));

rt = raytracer(c,2)

def plot_background(xlo, xhi, ylo, yhi, nx, ny):
	xx = np.linspace(xlo, xhi, nx);
	yy = np.linspace(ylo, yhi, ny);
	pp = np.array([[x,y] for x in xx for y in yy]);
	zz = np.zeros(nx * ny);
	for i in range(nx * ny):
		zz[i] = c(pp[i])
	zz = np.reshape(zz,(nx,ny))
	zz = np.flipud(zz.T)
	fig, ax = plt.subplots()
	cax = ax.imshow(zz, extent=[xlo, xhi, ylo, yhi], cmap = plt.get_cmap('Blues_r'));
	fig.colorbar(cax, fraction=0.046, pad=0.04, aspect=7.25)
	ax.axis([xlo, xhi, ylo, yhi]);
	ax.set_ylim([yhi, ylo]);
	return fig, ax

def make_normals(xlo, xhi, T, n_rays, n_steps):
	x0s = np.linspace(xlo, xhi, n_rays);
	x0s = np.array([[x0,0.0] for x0 in x0s]);
	xi0 = np.array([0.0,1.0]);

	data = []
	for x0 in x0s:
		xs,tt = rt.solve_hamilton_jacobi(x0, xi0, n_steps, T);
		datum = {'xs' :xs, 'tt':tt};
		data.append(datum);
	return data

def add_normals_to_plot(fig, ax, T, n_rays, n_steps, xlo, xhi):
	data = make_normals(xlo, xhi, T, n_rays, n_steps);
	for datum in data:
		xs = datum['xs'];
		ax.plot(xs[:,0], xs[:,1], c='b');	
	return data

def add_fan_to_plot(fig, ax, x0, T, n_rays, n_steps, theta_lo = np.pi, theta_hi = 2.0 * np.pi):
	data = rt.fire_ray_fan(x0, n_rays, n_steps, T, theta_lo, theta_hi);

	for datum in data:
		xs = datum['xs']
		tt = datum['total_travel_time']
		ax.plot(xs[:,0], xs[:,1], c='k');
	return data

def make_endpt_fan(x0, xi0, T, n_rays, n_steps, theta_lo = np.pi, theta_hi = 2.0 * np.pi):
	xs, tt = rt.solve_hamilton_jacobi(x0, xi0, n_steps, T);
	data = rt.fire_ray_fan(xs[-1,:], n_rays, n_steps,  tt, theta_lo, theta_hi);
	return data, xs

def add_endpt_fan_to_plot(fig, ax, x0, xi0, T, n_rays, n_steps, theta_lo = np.pi, theta_hi = 2.0 * np.pi):
	data, xs = make_endpt_fan(x0, xi0, T, n_rays, n_steps, theta_lo = np.pi, theta_hi = 2.0 * np.pi);

	ax.plot(xs[:,0], xs[:,1]);
	for datum in data:
		xs = datum['xs']
		tt = datum['total_travel_time']
		ax.plot(xs[:,0], xs[:,1], c='r');
	return data, xs

def make_isochrones(data, n_steps):
	x_data = np.zeros((n_steps, len(data)))
 	y_data = np.zeros((n_steps, len(data)))

 	for j, datum in enumerate(data):
 		xs = datum['xs']
 		x_data[:, j] = xs[:,0]
 		y_data[:, j] = xs[:,1]
	return x_data, y_data

def add_isochrones_to_plot(fig, ax, data, n_steps, stride):
	x_data, y_data = make_isochrones(data, n_steps);

	for i in np.arange(0, n_steps, stride):
		x_level = x_data[i,:]
		y_level = y_data[i,:]
		ax.plot(x_level, y_level, c='g')

if __name__ == '__main__':
	fig,ax = plot_background(-2.0, 2.0, 0.0, 1.25, 100,100);	
	fig.savefig('./img/TRIP_model.png', bbox_inches='tight')

	fig,ax = plot_background(-1.5, 1.5, 0.0, 1.25, 100, 100);	
	T = .75
	n_rays = 61
	n_steps = 101
	data = add_normals_to_plot(fig, ax, T, n_rays, n_steps, -1.5, 1.5);
	
	# # save:
	# x_rt = np.zeros((n_steps, n_rays))
	# y_rt = np.zeros((n_steps, n_rays))
	# for i, datum in enumerate(data):
	# 	x_rt[:,i] = datum['xs'][:,0];
	# 	y_rt[:,i] = datum['xs'][:,1];		
	# sio.savemat('./coords.mat', {'x_rt' : x_rt, 'y_rt' : y_rt})

	add_isochrones_to_plot(fig, ax, data, n_steps, 4);
	fig.savefig('./img/TRIP_BNC.png', bbox_inches='tight')
	
	x0 = np.array([-.85, 0.0]);
	#xi0 = np.array([1, 1.25]);
	T = 2.0;
	n_rays = 11;
	n_steps = 100;
	theta_lo, theta_hi =  np.pi/6, np.pi/3
	fig,ax = plot_background(-1.5, 1.5, 0.0, 1.25, 100, 100);
	data = add_fan_to_plot(fig, ax, x0, T, n_rays, n_steps, theta_lo, theta_hi)
	fig.savefig('./img/TRIP_Replica.png', bbox_inches='tight')

	plt.show();
