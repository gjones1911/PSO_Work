import numpy as np
import pandas as pd
import os
import sys
from P_SWARMS.Swarms import *

# method that processes the comand line and sets parameters for the swarm
# accordingly
def handle_cmd_line(particles=30, inertia=.5, cognit=2, soc=2, epochs=2000,
                    peers=0, decay=0, rate=10, objective=1,):
    if len(sys.argv) > 1:
        num_part = int(sys.argv[1])
        inertia = float(sys.argv[2])
        cognition = int(sys.argv[3])
        social = int(sys.argv[4])
        epochs = int(sys.argv[5])
        objective = int(sys.argv[6])
        if len(sys.argv) > 7:
            decay = float(sys.argv[7])
        if len(sys.argv) > 8:
            rate = float(sys.argv[8])
        if len(sys.argv) > 9:
            peers = float(sys.argv[9])
        return num_part, inertia, cognition, social, epochs, objective, decay, rate, peers,
    else:
        num_part = particles    # 1
        inertia = inertia       # 2
        cognition = cognit      # 3
        social = soc            # 4
        epochs = epochs         # 5
        objective = objective   # 6
        decay = decay           # 7
        rate = rate             # 8
        peers = peers           # 9
        return num_part, inertia, cognition, social, epochs, objective, decay, rate, peers,


show_it = True           # determines if graphs or displayed for results
save_it = False          # determines if reports are saved
save_plots = False       # determines if the plots are saved
num_part, inertia, cognit, soc, epochs, objective, decay, rate, peers = handle_cmd_line()


if len(sys.argv) > 10:
    t_type = float(sys.argv[10])
else:
    t_type = 0
print('Particles: {}, inertia: {}, cognition: {}, social: {}, epochs: {}, objective: {}, decay: {}, rate: {}, peers: {}'.format(
    num_part, inertia, cognit, soc,  epochs, objective, decay, rate, peers,
))
print('Performing testing type: {}'.format(t_type))
convergence_thresh=.001
error_threshold = -np.inf
loss_threshold = -np.inf
xy_thesh = .0000001
xy_thesh = -np.inf              # threshold for the x and y loss,
min_inert = .01                 # minimum inertia for inertia adjustment
ajnumber = 1                    # determines type of adjustment leave at 1
adj_inert=False                 # determines if inertia adjustment is to occur
vel_update=False                # determines if k nearest neighbors will be used to update the velocity
if decay != 0:
    adj_inert = True
else:
    adj_inert = False
if peers > 0:
    vel_update = True
else:
    vel_update = False
print('I was given {} as the testing type'.format(t_type))

# create the world for and generate the swarm with provided paramters
swarm = Swarm(num_part=num_part, inertia=inertia, cognit=cognit, soc_pres=soc, min_inert=min_inert,
              vel_update_mode=vel_update, epochs=epochs, error_threshold=error_threshold, obj=objective,
              convergence_thresh=convergence_thresh, decay=decay, rate=rate, adj_inert=adj_inert,
              peer=peers, loss_thresh=xy_thesh, adjuster_num=ajnumber, save_plots=save_plots, testing_type=t_type)

print('Testing type {}'.format(swarm.test_dirs_dict[t_type]))
# start particle swarm optimization
swarm.start_swarm()

# once the swarm completes optionally save the data as a csv and/or the plots as png's
if save_it:
    # TODO: plot the stored average performance for this set of parameters
    swarm.save_report()
    swarm.plot_report_files(show_it=False)
    swarm.plot_final_particle_positions(show_it=False)
# TODO: once the swarm has stopped

# TODO: store the results for this set of parameters

if show_it:
    # TODO: display the initial positions of the particles
    swarm.plot_intial()

    # TODO: display the final positions of the particles, their best found solution, and the true solution
    swarm.plot_final_particle_positions()
    swarm.plot_average_loss()
    swarm.plot_xyaverage_loss()
    #swarm.plot_report_files(show_it=show_it, range=None)
    plt.show()



