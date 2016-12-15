import numpy as np
import matplotlib.pyplot as plt
from raytracer import raytracer

def c(x):
	return (2.0 + x[1]
			- 1.5 * np.exp( -4.0 * ((x[0] * x[0]) + (x[1] - 1.5) * (x[1] - 1.5))))

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
	fig.colorbar(cax,  shrink=.5, pad=.2, aspect=10)
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
	fig,ax = plot_background(-4.0, 4.0, 0.0, 3.0, 100,100);	
	data = add_normals_to_plot(fig, ax, 2.0, 31, 200, -4.0, 4.0);
	add_isochrones_to_plot(fig, ax, data, 200, 3);
	fig.savefig('./img/TRIP_BNC.png', bbox_inches='tight')

	x0 = np.array([-3.5, 0.0]);
	xi0 = np.array([1, 1.25]);
	T = 4.0;
	n_rays = 31;
	n_steps = 100;
	theta_lo, theta_hi =  0.7, 1.15
	fig,ax = plot_background(-4.0, 4.0, 0.0 ,3.0,100,100);
	data = add_fan_to_plot(fig, ax, x0, T, n_rays, n_steps, theta_lo, theta_hi)
	fig.savefig('./img/TRIP_Reeplica.png', bbox_inches='tight')

	plt.show();
