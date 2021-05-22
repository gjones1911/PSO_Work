import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
def usage():
    print('usage: epochs particles inertia cognition social peer* decay* rate*')
    print('usage: epochs particles inertia cognition social peer* decay* rate*')
    print('usage: python exe.py')

def handle_cmd_line(particles=20, inertia=.2, cognit=2, soc=2, epochs=5000, peers=0, decay=.5, rate=50, objective=1,
                    ):
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        num_part = int(sys.argv[2])
        inertia = float(sys.argv[3])
        cognition = int(sys.argv[4])
        social = int(sys.argv[5])
        objective = int(sys.argv[6])
        if len(sys.argv) > 7:
            decay = float(sys.argv[7])
        if len(sys.argv) > 8:
            rate = float(sys.argv[8])
        if len(sys.argv) > 9:
            peers = float(sys.argv[9])
        return num_part, inertia, cognition, social, epochs, objective, decay, rate, peers,
    else:
        epocs = epochs         # 5
        num_part = particles    # 1
        inertia = inertia       # 2
        cognition = cognit     # 3
        social = soc            # 4
        objective = objective   # 6
        decay = decay           # 7
        rate = rate             # 8
        peers = peers           # 9
        return num_part, inertia, cognition, social, epochs, objective, decay, rate, peers,

def inert_Adjuster1(test_count, rate, inertia, decay, min_inert):
    print('INERTial adjuster 1')
    # TODO: the higher the decay the slower the decay
    # TODO: the lower the decay the faster the decay
    if test_count % rate == 0:
        return max(inertia * decay, min_inert)
    else:
        return inertia


def inert_Adjuster2(test_count, rate, inertia, decay, min_inert):
    print('INERTIal adjuster 2')
    # TODO: the higher the decay the quicker the decay
    # TODO: the lower the decay the slower the decay
    if test_count % rate == 0:
        return max(inertia - (inertia * decay), min_inert)
    else:
        return inertia

def generate_ineria_adjustment_plots(adj_num, tests, rate, inertia, decay, min_inert,):
    adjuster = None
    if adj_num == 1:
        adjuster = inert_Adjuster1
    elif adj_num == 2:
        adjuster = inert_Adjuster2

    y = list()
    x = list()
    for test_count in range(tests):
        y.append(inertia)
        x.append(test_count+1)
        inertia =adjuster(test_count, rate, inertia, decay, min_inert)
    return x, y

def display_objectives_space(show_the_objectives, adjnm, minrt, dcy, epochs, inertia,
                             rate, ):
    if show_the_objectives:
        #adjnm = 1
        #minrt = min_inert
        #dcy = decay
        if adjnm == 1 or adjnm == 3:
            x1, y1 = generate_ineria_adjustment_plots(adjnm, tests = epochs, inertia=inertia, decay=dcy,
                                                      rate=rate, min_inert=minrt)
            plt.figure()
            plt.plot(x1, y1)
            plt.title('Adjuster {}: decay = {}, rate = {}, inertia = {}'.format(adjnm, dcy, rate, inertia))
            plt.show()
            print('---------------------------')
            print('---------------------------')
            print('---------------------------')
            print('---------------------------')
            print('---------------------------')
        if adjnm == 2 or adjnm == 3:
            x2, y2 = generate_ineria_adjustment_plots(adjnm, tests = epochs, inertia=inertia,
                                                      decay=dcy, rate=rate, min_inert=minrt)
            plt.figure()
            plt.plot(x2, y2)
            plt.title('Adjuster {}: decay = {}, rate = {}, inertia = {}'.format(adjnm, dcy, rate, inertia))
            plt.show()

"""
def handle_cmd_line(epochs=10, num_part=20, inertia=.89, cognit=2, soc_press=2):
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])       # number of epochs
        num_part = int(sys.argv[2])     # number of particles
        inertia = float(sys.argv[3])    # inertia
        cognit = int(sys.argv[4])       # cognition parameter
        soc_press = int(sys.argv[5])    # social parameter
    return epochs, num_part, inertia, cognit, soc_press
"""

def get_normal_val():
    return np.random.uniform(0, 1)

def plot_funct(objective, xy, dims):
    img = np.zeros(dims)
    #for rows, x in zip(list(range(len(xy[0]))), xy[0]):
    #    for cols, y in zip(list(range(len(xy[1]))), xy[1]):
    max_val = 0
    max_cords = [0,0]
    for x in xy[0]:
        for y in xy[1]:
            img[x+50-1][y+50-1] = objective(x, y)
            if img[x+50-1][y+50-1] > max_val:
                max_val = img[x+50-1][y+50-1]
                max_cords[0] =x+50-1
                max_cords[1] =y+50-1

    plt.figure()
    plt.text(max_cords[0]-10, max_cords[1]+22, 'Global Maximum', )
    #plt.text(-(max_cords[0]-50+1)+10, -(max_cords[1]-50+1)+15, 'Global Maximum')
    #plt.text()
    plt.title('Objective 1:\nmax value of {} is at {}'.format(np.around(max_val, 2),
                                                                    (max_cords[0]-50+1, max_cords[1]-50+1),))
                                                                   #(-(max_cords[0] - 50 + 1),-(max_cords[1] - 50 + 1))))
    plt.imshow(img, cmap='magma')
    plt.yticks(list(range(100)),['']*len(list(range(-50, 50+1, 1))),)
    plt.xticks(list(range(100)),['']*len(list(range(-50, 50+1, 1))),)
    #plt.xticks(list(range(100)), list(range(-50, 50+1, 1)))
    #plt.yticks(list(range(100)), list(range(-50, 50+1, 1)).reverse()
    plt.show()

class Swarm:
    solnX, solnY = 20, 7  # true solution to objectives/problems
    # particles reprsented by array of the form
    # [x, y, vel, best, global_best, last_fit]
    #report_file_name = '_Experiment_Files/_P{}/Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}.csv'
    #report_file_name_con = '_Experiment_Files/_P{}/SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}.csv'

    #report_file_name = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}\_data_files\{}\Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    #report_file_name_con = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}\_data_files\{}\SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    report_file_name = r'_P{}_{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    report_file_name_con = r'_P{}_{}_SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    #report_file_name_con = r'Result_Plots\_P{}\_data_files\{}\SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'

    #report_file_name2 = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}_2\_data_files\{}\Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    #report_file_name_con2 = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}_2\_data_files\{}\SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    report_file_name2 = r'_P{}_2_{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'
    report_file_name_con2 = r'_P{}_2_{}_SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.csv'

    #plot_file_name_ = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}\_plots\{}\{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    #plot_file_name_con = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}\_plots\{}\SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    #plot_file_name_final_pos = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}\_plots\{}\FinalPos_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'

    plot_file_name_ = r'_P{}_plots_{}_{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    plot_file_name_con = r'_P{}_plots_{}_SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    plot_file_name_final_pos = r'_P{}_plots_{}_FinalPos_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'

    #plot_file_name2_ = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}_2\_plots\{}\{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    #plot_file_name_con2 = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}_2\_plots\{}\SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    #plot_file_name_final_pos2 = r'C:\Users\gjone\BIO_COMP_P5\_Experiment_Files\_P{}_2\_plots\{}\FinalPos_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'

    plot_file_name2_ = r'_P{}_2_plots_{}_{}_Swarm_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    plot_file_name_con2 = r'_P{}_2_plots_{}_SwarmCon_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'
    plot_file_name_final_pos2 = r'_P{}_2_plots_{}_FinalPos_inrt{}_cog{}_soc{}_dcy{}_r{}_p{}_ep{}.png'

    base_title_string = '{:s} Objective {:d}\nP: {}, World: {}, Inertia: {}, Cognition: {} Social: {}'
    #base_xlabel_string = 'Epoch\nInertia Decay: {:.4f}, Decay Interval: {:d}, Peer Pressure: {:.4f}, k={}'
    base_xlabel_string = 'Epoch\nInertia Decay: {}, Decay Interval: {}, Peer Pressure: {}, k={}'
    base_xlabel_string = 'Inertia Decay: {}, Decay Interval: {}, Peer Pressure: {}, k={}'
    avg_plot_title = 'Avg. Loss/Epoch:'
    percent_converged_title = '% of {} Particles that Converged to Solution:'
    final_pos_title = 'Final Particle Positions:'
    initial_pos_title = 'Initial Particle Positions:'
    xy_avg_loss_title = 'Average X/Y loss:'
    loss_img_name = 'Avg_loss_'
    xyloss_img_name = 'Avg_XY_loss_'
    inert_adj_test_str = 'inertia_adjustment_tests'
    cognition_test_str = 'cognition_tests'
    inert_test_str = 'inertia_tests'
    nearest_test_str = 'nearest_peers_tests'
    soc_test_str = 'social_tests'

    test_dirs_dict = {
                      0:inert_test_str,
                      1:cognition_test_str,
                      2:soc_test_str,
                      3:inert_adj_test_str,
                      4:nearest_test_str,
    }

    def __init__(self, num_part, inertia=.5, cognit=2, world_shape=(100, 100), soc_pres=2, max_vel=1, verbose=False, min_inert=.5,
                 obj=1, epochs=500, error_threshold=.01, convergence_thresh =.05, decay = .99, rate=1, adj_inert=False,
                 peer=1.5, k=2, vel_update_mode=False, loss_thresh=.0001, adjuster_num=1, testing_type=0, save_plots=False):
        self.verbose = verbose
        self.save_plots = save_plots
        self.testing_type = testing_type
        self.avg_loss_plot_img = None
        self.xy_loss_plot_img = None
        self.convg_plot_img = None
        self.final_pos_plot_img = None
        # TODO: World specs
        self.world_shape = world_shape
        self.height_dim = int(np.ceil(world_shape[0]/2))
        self.width_dim = int(np.ceil(world_shape[1]/2))

        # TODO: particle specs
        self.num_part = num_part
        self.inertia = inertia
        self.cognition = cognit
        self.social_pressures = soc_pres
        self.peer = peer

        # TODO: Learning specs
        # store which objective function is being used for this run
        self.obj = obj                          # used to determine which objective function to use
        self.objective = self.set_objective(self.obj) # set objective function
        self.decay = decay
        self.vel_update_mode = vel_update_mode
        self.rate = rate                              #determins the epoch were decay is applied (*= decay)
        self.epochs = epochs
        self.adjust_inert = adj_inert                  # used to determine if inertia is adaptive
        self.adjuster = self.set_adjuster(adjuster_num)
        self.adjuster_num = adjuster_num
        self.error_thresh = error_threshold            # used to for avg loss in x and y
        self.loss_thresh = loss_thresh                 # used for x and y loss
        if not adj_inert:
            self.min_inert = self.inertia
        elif adj_inert:
            self.min_inert = min_inert
        self.k = k
        #if self.adjust_inert:
        #    adp = 'True'
        #else:
        #    adp = 'False'
        #if self.vel_update_mode == 0:
        #    k = 'None'
        #else:
        #    k = str(self.k)
        # want most dispersion possible/cast wide net
        self.init_posX = [x for x in np.random.choice(list(range(-self.height_dim, self.height_dim+1, 1)), self.num_part, replace=False)]
        self.init_posY = [x for x in np.random.choice(list(range(-self.width_dim, self.width_dim+1, 1)), self.num_part, replace=False)]

        # TODO: Shared Info dump
        self.group_best = None              # store the groups best found fitness
        self.cnvg_cnt = 0                   # used to store how many converged on a given epoch
        self.convergence_threshold = convergence_thresh    #
        self.percentage_converge = list()                   # used to store the %converged each epoch
        self.max_velocity = max_vel                         # Used to contral the maximum velocity of the particles

        # TODO: the values below represent different information in the particle's array of info
        self.px, self.py = 0, 1                 # position in a particles data array for x, and y
        self.v = 2                              # index for velocity x and y values of particle ([x,y])
        self.bst = 3                            # index to stored bext solution array ([x,y])
        self.g_bst = 4                          # index to global best solution array ([x,y])
        self.b_val = 5                          # index to the particles best fitness value (objective func value)
        self.last_fit = 6                       # index the last fitness of the particle
        self.global_best = list([0,0])          # easy access to the current best global x and y values
        self.best_fit = 0                       # current best fitness score over all

        # TODO: set up the xmax, and ymax for calculation
        self.x_max = self.world_shape[1]        # set the max boundary for the world for calculation  scaleing purposes
        self.y_max = self.world_shape[0]        # same as above for y

        # set up variables for the plots
        self.losses, self.xlosses, self.ylosses = list(), list(), list()        # keeps track of loss per epoch
        self.tests = list()            # used to store the number of tests run, for plotting
        self.test_count = 1             # keeps track of curren test number
        self.avg_error = 900            # tracks the average error in the current system in x and y from soln
        self.errorY = 0                 # tracks the error in the x and y from the objective solution
        self.errorX = 0                 # tracks the error in the x and y from the objective solution
        self.loss_plot_title = ''
        self.xyloss_plot_title = ''
        self.convergence_plot_title = ''
        self.scatter_plot_title = ''
        self.initial_position_title = ''
        self.xlabel = ''
        if obj == 2:
            print('testing mode 2')
            self.set_plot_titles2()
        else:
            print('testing mode 1')
            self.set_plot_titles()
        # self.particles = self.generate_particles()   # list of list representing particles
        self.orig_best = self.calculate_fitness(mode=0) # best from the initial set up
        self.global_best = [self.orig_best[0], self.orig_best[1]]   # make a copy for learning
        self.particles = self.generate_particles()   # list of list representing particles # generate the particles


    def set_adjuster(self, adj_num):
        if adj_num == 1:
            return self.inert_Adjuster1
        elif adj_num == 2:
            return self.inert_Adjuster2
        else:
            return self.inert_Adjuster1

    def generate_particles(self,):
        """Used to create the list of particle arrays"""
        p_ret = list()
        for x, y in zip(self.init_posX, self.init_posY):
                        # x  y    v     bst   gbst              b_val last fit
            p_ret.append([x, y, [0,0], [x,y], self.global_best, 0,     0])
        return p_ret

    def set_plot_titles(self,):
        self.loss_plot_title= self.base_title_string.format(self.avg_plot_title, self.obj, self.num_part,
                                                       self.world_shape,self.inertia, self.cognition,
                                                       self.social_pressures)
        self.xyloss_plot_title = self.base_title_string.format(self.xy_avg_loss_title, self.obj, self.num_part,
                                                             self.world_shape, self.inertia, self.cognition,
                                                             self.social_pressures)

        self.convergence_plot_title = self.base_title_string.format(self.percent_converged_title.format(self.num_part), self.obj, self.num_part,
                                                       self.world_shape, self.inertia, self.cognition,
                                                       self.social_pressures)
        self.scatter_plot_title = self.base_title_string.format(self.final_pos_title, self.obj, self.num_part,
                                                           self.world_shape, self.inertia, self.cognition,
                                                           self.social_pressures)
        self.initial_position_title = self.base_title_string.format(self.initial_pos_title, self.obj, self.num_part,
                                                             self.world_shape, self.inertia, self.cognition,
                                                             self.social_pressures)
        rt, dc, prr, kv = 0, 0, 0, 0
        if self.adjust_inert:
           rt = np.around(self.rate, 4)
           dc = np.around(self.decay, 4)
        else:
            rt = None
            dc = None
        if self.vel_update_mode:
            prr = np.around(self.peer, 4)
            kv = self.k
        else:
            prr = None
            kv = None
        self.report_file_name = self.report_file_name.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             self.inertia, self.cognition,
                                                             self.social_pressures, str(dc), str(rt), str(prr), self.epochs)
        self.report_file_name_con = self.report_file_name_con.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                                     self.inertia, self.cognition,
                                                                     self.social_pressures, str(dc), str(rt), str(prr),
                                                                     self.epochs)


        self.avg_loss_plot_img = self.plot_file_name_.format(self.num_part, self.test_dirs_dict[self.testing_type], self.loss_img_name,
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                             self.epochs)
        self.xy_loss_plot_img = self.plot_file_name_.format(self.num_part, self.test_dirs_dict[self.testing_type],self.xyloss_img_name,
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                            self.epochs)
        self.convg_plot_img = self.plot_file_name_con.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                             self.epochs)
        self.final_pos_plot_img = self.plot_file_name_final_pos.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                                       self.epochs)


        self.xlabel = self.base_xlabel_string.format(dc, rt, prr, kv)

    def set_plot_titles2(self,):
        self.loss_plot_title= self.base_title_string.format(self.avg_plot_title, self.obj, self.num_part,
                                                       self.world_shape,self.inertia, self.cognition,
                                                       self.social_pressures)
        self.xyloss_plot_title = self.base_title_string.format(self.xy_avg_loss_title, self.obj, self.num_part,
                                                             self.world_shape, self.inertia, self.cognition,
                                                             self.social_pressures)
        self.convergence_plot_title = self.base_title_string.format(self.percent_converged_title.format(self.num_part), self.obj, self.num_part,
                                                       self.world_shape, self.inertia, self.cognition,
                                                       self.social_pressures)
        self.scatter_plot_title = self.base_title_string.format(self.final_pos_title, self.obj, self.num_part,
                                                           self.world_shape, self.inertia, self.cognition,
                                                           self.social_pressures)
        self.initial_position_title = self.base_title_string.format(self.initial_pos_title, self.obj, self.num_part,
                                                             self.world_shape, self.inertia, self.cognition,
                                                             self.social_pressures)
        rt, dc, prr, kv = 0, 0, 0, 0
        if self.adjust_inert:
           rt = np.around(self.rate, 4)
           dc = np.around(self.decay, 4)
        else:
            rt = None
            dc = None
        if self.vel_update_mode:
            prr = np.around(self.peer, 4)
            kv = self.k
        else:
            prr = None
            kv = None
        self.report_file_name = self.report_file_name2.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             self.inertia, self.cognition,
                                                             self.social_pressures, str(dc), str(rt), str(prr), self.epochs)
        self.report_file_name_con = self.report_file_name_con2.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                                     self.inertia, self.cognition,
                                                                     self.social_pressures, str(dc), str(rt), str(prr),
                                                                     self.epochs)
        self.avg_loss_plot_img = self.plot_file_name2_.format(self.num_part, self.test_dirs_dict[self.testing_type], self.loss_img_name,
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                             self.epochs)
        self.xy_loss_plot_img = self.plot_file_name2_.format(self.num_part, self.test_dirs_dict[self.testing_type],self.xyloss_img_name,
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                            self.epochs)
        self.convg_plot_img = self.plot_file_name_con2.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                             self.epochs)
        self.final_pos_plot_img = self.plot_file_name_final_pos2.format(self.num_part, self.test_dirs_dict[self.testing_type],
                                                             np.around(self.inertia, 2), self.cognition,
                                                             self.social_pressures, self.decay, self.rate, self.peer,
                                                                       self.epochs)
        self.xlabel = self.base_xlabel_string.format(dc, rt, prr, kv)

    def set_objective(self, obj):
        if obj is None:
            print('Using Objective function 1')
            obj = self.obj      # default to objective 1

        if obj == 1:
            print('Using Objective function 1')
            return self.obj_1
        else:
            print('Using Objective function 2')
            return self.obj_2

    def get_global_best_val(self, obj=None):
        """
            Just returns the current global best value
        :param obj:
        :return:
        """
        return self.objective(self.global_best[0], self.global_best[1])

    def inert_Adjuster1(self):
        print('inertial adjuster 1')
        # TODO: the higher the decay value the slower the decay
        if self.test_count % self.rate == 0:
            return max(self.inertia * self.decay, self.min_inert)
        else:
            return self.inertia

    def inert_Adjuster2(self):
        print('inertial adjuster 2')
        # TODO: the higher the decay the quicker the decay
        if self.test_count % self.rate == 0:
            return max(self.inertia - self.inertia*self.decay, self.min_inert)
        else:
            return self.inertia

    def adjust_inertia(self, mode=0):
        if self.adjust_inert:
            if mode == 0:
                #self.inertia = self.inert_Adjuster1()
                self.inertia = self.adjuster()

    def calculate_pfit(self, objective):
        """ calculates the fitness of each particle at their current postion and finds/keeps track the personel/global best"""
        #print('initial global best: {}, at {}'.format(self.global_best, self.best_fit))
        for particle in self.particles:
            c_fit = objective(particle[0], particle[1])
            particle[self.last_fit] = c_fit     # store the last fitness score
            # calculate the fit of this particle
            # as it is and if it is better at this
            # state store it and replace the old
            #if c_fit >= particle[self.b_val]:
            if c_fit >= particle[self.b_val]:
                particle[self.b_val] = c_fit
                particle[self.bst][0] = particle[0]
                particle[self.bst][1] = particle[1]
            # do above but for best global fit
            #if self.best_fit <= c_fit:
            if self.best_fit <= c_fit:
                #print('new global best: {}'.format(c_fit))
                self.best_fit = c_fit
                self.global_best[0] = particle[0]
                self.global_best[1] = particle[1]
        #print('best fit is now at: {}, with a fit value of {}'.format(self.global_best, self.best_fit))
        self.set_best_global_fitness()
        return

    def set_best_global_fitness(self, ):
        for particle in self.particles:
            particle[self.g_bst] = self.global_best

    def best_loss(self):
        """
            Calculates and stores the current average loss
        :return:
        """
        errorX, errorY = 0,0
        for particle in self.particles:
            #errorX += (particle[self.bst][0] - self.global_best[0])**2
            #errorY += (particle[self.bst][1] - self.global_best[1])**2
            errorX += (particle[self.bst][0] - self.solnX)**2
            errorY += (particle[self.bst][1] - self.solnY)**2

        self.errorX = np.sqrt( (1/(2*self.num_part)) * errorX)
        self.errorY = np.sqrt( (1/(2*self.num_part)) * errorY)
        self.avg_error = (self.errorX + self.errorY)/2
        self.losses.append(self.avg_error)
        self.xlosses.append(self.errorX)
        self.ylosses.append(self.errorY)
        if self.verbose:
            print('error for epoch {:d}: {:.3f}, current best: [{:d}, {:d}], {:.2f}'.format(self.test_count,
                                                                                            self.avg_error,
                                                                                            self.global_best[0],
                                                                                            self.global_best[1],
                                                                                            self.get_global_best_val(self.global_best[0], self.global_best[1])))

    def get_best_neighbor(self, idx):
        l_n = self.particles[idx-1]
        me = self.particles[idx]
        h_n = self.particles[(idx+1)%self.num_part]
        neighbors = [l_n, me, h_n]
        mxx = [l_n[self.last_fit], me[self.last_fit], h_n[self.last_fit]]
        return neighbors[mxx.index(max(mxx))]

    def update_velocity_pos(self, mode=0):
        self.cnvg_cnt = 0
        idx = 0
        for particle in self.particles:
            if mode == 0:
                particle[self.v][0] = self.calculate_velocity(particle[self.v][0], particle[self.bst][0], particle[self.px], self.global_best[0])  # x vel
                particle[self.v][1] = self.calculate_velocity(particle[self.v][1], particle[self.bst][1], particle[self.py], self.global_best[1])  # yvel
            else:
                # get the best local neighbor(idx)
                #print('looking at the neighbors')
                bnn = self.get_best_neighbor(idx)
                particle[self.v][0] = self.calculate_velocityTribe(particle[self.v][0], particle[self.bst][0],
                                                              particle[self.px], self.global_best[0], bnn[self.bst][0])  # x vel
                particle[self.v][1] = self.calculate_velocityTribe(particle[self.v][1], particle[self.bst][1], particle[self.py],
                                                          self.global_best[1], bnn[self.bst][1])  # yvel

            if (particle[self.v][0]**2 + particle[self.v][1]**2) > self.max_velocity**2:
                denom = np.sqrt(particle[self.v][0]**2 + particle[self.v][1]**2)
                particle[self.v][0] = (self.max_velocity/(denom)) * particle[self.v][0]
                particle[self.v][1] = (self.max_velocity/(denom)) * particle[self.v][1]
            # use new velocity update position
            new_x = particle[0] + particle[self.v][0]
            new_y = particle[1] + particle[self.v][1]
            sgn = 1
            if new_x < 0:
                #new_x = min(new_x, self.max)
                new_x *= -1
                new_x = -1*(new_x%self.width_dim)
            else:
                new_x = (new_x%self.width_dim)

            if new_y < 0:
                new_y *= -1
                new_y = -1*(new_y%self.height_dim)
            else:
                new_y = (new_y%self.height_dim)

            #particle[0] += particle[self.v][0]
            #particle[1] += particle[self.v][1]
            particle[0] = new_x
            particle[1] = new_y
            if self.convergence_threshold > np.linalg.norm(np.array([particle[0], particle[1]])-np.array([self.solnX, self.solnY])):
                self.cnvg_cnt += 1
            idx += 1
        self.percentage_converge.append(np.around((self.cnvg_cnt/self.num_part), 6))
        return

    def calculate_velocity(self, cvel, m_best, pos, g_best,):
        # Leader, follower, pragmatic
        # drive, ambition,
        # lazyness apathy
        # where should the particle head?
        momentum = self.inertia * cvel
        cognitive_weight = self.cognition * get_normal_val()            # my logic, experiences and feeling toward what is best
        social_weight = self.social_pressures * get_normal_val()        # what society/environment tells me is best
        dif_m_best = m_best - pos                                       # difference between where I am and what I think is best
        dif_g_best = g_best - pos                                       # difference between where I am and where society/environment says is best
        # current direction + where I know is best weighted my my knowldege/thoughts + the environment/society * where the society says is best
        return momentum + cognitive_weight*dif_m_best + social_weight*dif_g_best

    def calculate_velocityTribe(self, cvel, m_best, pos, g_best, bnn_best):
        # Leader, follower, pragmatic
        # drive, ambition,
        # lazyness apathy
        # where should the particle head?
        peer_weight = self.peer * get_normal_val()                      # nearest peers influence
        momentum = self.inertia * cvel                                  # weight give to current velocity
        cognitive_weight = self.cognition * get_normal_val()            # my logic, experiences and feeling toward what is best
        social_weight = self.social_pressures * get_normal_val()        # what society/environment tells me is best
        dif_m_best = m_best - pos                                       # difference between where I am and what I think is best
        dif_g_best = g_best - pos                                       # difference between where I am and where society/environment says is best
        dif_bn_best = bnn_best - pos                                    # difference between where I am and where my closest particles are
        # current direction + where I know is best weighted my my knowldege/thoughts + the environment/society * where the society says is best
        return momentum + cognitive_weight*dif_m_best + social_weight*dif_g_best + peer_weight * dif_bn_best

    def start_swarm(self):
        self.train(self.vel_update_mode)
    def train(self, mode=0):
        c_err = 900
        ep = 0
        print('training mode', mode)
        while self.avg_error > self.error_thresh and ep < self.epochs:
            # calculate loss
            #print('error ', self.avg_error)
            self.calculate_fitness(mode=1)      # get the calculate fitness
            self.best_loss()                    # calculate the average loss

            # update velocity and position of particles
            if mode == False:
                self.update_velocity_pos()
            else:
                #print('vfancy velocity')
                self.update_velocity_pos(mode=mode)
            self.adjust_inertia()
            if self.test_count%self.rate == 0:
                print('-----------------: epoch {:d}: avg_loss: {:.3f}, inertia: {:.3f}'.format(int(ep), np.around(self.avg_error, 2), self.inertia))
                print('-----------------: x and y loss: {}, {}'.format(self.errorX, self.errorY))
            self.test_count += 1
            ep += 1
            # update
            if self.errorX < self.loss_thresh and self.errorY < self.loss_thresh:
                print('x and y loss threshold ({}) met, x:{}, y:{}'.format(self.loss_thresh, self.errorX, self.errorY))
                return
        if self.avg_error < self.error_thresh:
            print('Average loss threshold met: {}'.format(self.avg_error))
        else:
            print('Training Epoch limit met')

    def mdist(self,):
        xsqr, ysqr = self.x_max**2, self.y_max**2
        return np.sqrt(xsqr + ysqr)/2
    def pdist(self, px, py,):
        a = (px - 20)**2
        b = (py - 7)**2
        return np.sqrt(a + b)
    def ndist(self, px, py, xsub=20, ysub=7):
        a = (px + xsub) ** 2
        b = (py + ysub) ** 2
        return np.sqrt(a + b)

    def calculate_fitness(self, func=0, mode=0):
        if mode == 0:
            return self.calculate_initial_fit(self.objective)
        else:
            return self.calculate_pfit(self.objective)
    def calculate_initial_fit(self, objective):
        max_fit = 0
        b_cords = [0,0]
        for x, y in zip(self.init_posX, self.init_posY):
            c_fit = objective(x, y)
            if c_fit > max_fit:
                max_fit = c_fit
                b_cords[0] = x
                b_cords[1] = y
        return b_cords

    def obj_1(self, x, y):
        return 100 * (1 - (self.pdist(x, y)/self.mdist()))
    def obj_2(self, x, y):
        return 9 * max(0, 10 - self.pdist(x, y)**2) + 10*(1-(self.pdist(x, y)/self.mdist())) + 70*(1 - (self.ndist(x, y)/self.mdist()))

    def plot_funct(self, show_it=False):
        "plots objective functions"
        xy = list(range(-50, 51)), list(range(-50, 51))
        dims = (100, 100)

        img = np.zeros(dims)
        # for rows, x in zip(list(range(len(xy[0]))), xy[0]):
        #    for cols, y in zip(list(range(len(xy[1]))), xy[1]):
        max_val = 0
        max_cords = [0, 0]
        for x in xy[0]:
            for y in xy[1]:
                img[x + 50 - 1][y + 50 - 1] = self.objective(x, y)
                if img[x + 50 - 1][y + 50 - 1] > max_val:
                    max_val = img[x + 50 - 1][y + 50 - 1]
                    max_cords[0] = x + 50 - 1
                    max_cords[1] = y + 50 - 1

        plt.figure()
        plt.text(max_cords[0] - 10, max_cords[1] + 22, 'Global Maximum', )
        # plt.text(-(max_cords[0]-50+1)+10, -(max_cords[1]-50+1)+15, 'Global Maximum')
        # plt.text()
        plt.title('Objective 1:\nmax value of {} is at {}'.format(np.around(max_val, 2),
                                                                  (max_cords[0] - 50 + 1, max_cords[1] - 50 + 1), ))
        # (-(max_cords[0] - 50 + 1),-(max_cords[1] - 50 + 1))))
        plt.imshow(img, cmap='magma')
        plt.yticks(list(range(100)), [''] * len(list(range(-50, 50 + 1, 1))), )
        plt.xticks(list(range(100)), [''] * len(list(range(-50, 50 + 1, 1))), )
        # plt.xticks(list(range(100)), list(range(-50, 50+1, 1)))
        # plt.yticks(list(range(100)), list(range(-50, 50+1, 1)).reverse()
        plt.show()

    def plot_average_loss(self, show_it=False):
        plt.figure()
        plt.xlabel('Epochs')
        plt.xlabel(self.xlabel)
        plt.ylabel('average error')
        #plt.legend([])
        plt.title(self.loss_plot_title)
        plt.plot(range(1, len(self.losses)+1), self.losses)
        plt.legend(['final avg Loss: {:.4}'.format(self.losses[-1])], loc='best')

        if self.save_plots:
            print('saving the average loss plot at {}'.format(self.avg_loss_plot_img))
            plt.savefig(self.avg_loss_plot_img)

        if show_it:
            plt.show()
    def plot_xyaverage_loss(self, show_it=False):
        plt.figure()
        plt.xlabel('Epochs')
        plt.xlabel(self.xlabel)
        plt.ylabel('average error')
        #plt.legend([])
        plt.title(self.loss_plot_title)
        plt.plot(range(1, len(self.xlosses)+1), self.xlosses)
        plt.plot(range(1, len(self.ylosses)+1), self.ylosses)
        plt.legend(['Final X Loss: {:f}'.format(self.xlosses[-1]),
                    'Final Y loss: {:f}'.format(self.ylosses[-1])], loc='best')
        if self.save_plots:
            print('Saving the average X/Y loss plot at {}'.format(self.xy_loss_plot_img))
            plt.savefig(self.xy_loss_plot_img)
        if show_it:
            plt.show()
    def percentage_plot(self, show_it=False):
        plt.figure()
        plt.title(self.convergence_plot_title)
        plt.xlabel('Epoch')
        plt.xlabel(self.xlabel)
        plt.ylabel('Percentage of Converged Particles')
        plt.plot(range(1, len(self.percentage_converge)+1), self.percentage_converge)

        if self.save_plots:
            print('Saving the % Converged plot at {}'.format(self.convg_plot_img))
            plt.savefig(self.convg_plot_img)

        if show_it:
            plt.show()
    def plot_final_particle_positions(self, show_it=False):
        plt.figure()
        #plt.xlabel('Epoch')
        #plt.ylabel('')
        plt.title(self.scatter_plot_title)
        plt.xlabel(self.xlabel)
        plt.grid(True)
        x = [particle[self.px] for particle in self.particles]
        y = [particle[self.py] for particle in self.particles]
        #for particle in self.particles:
        plt.scatter(self.solnX, self.solnY, c='r', s=150)
        plt.scatter(self.global_best[0], self.global_best[1], c='y', s=100)
        plt.scatter(x, y, c='b')
        plt.legend(['solution', 'found best', 'particle'])

        if self.save_plots:
            print('Saving the final particle positions plot at {}'.format(self.final_pos_plot_img))
            plt.savefig(self.final_pos_plot_img)

        if show_it:
            plt.show()
    def plot_intial(self, show_it=False):
        plt.figure()
        plt.title('Initial positions')
        plt.xlabel(self.xlabel)
        plt.grid(True)
        x = [x for x in self.init_posX]
        y = [y for y in self.init_posY]
        #y = [particle[self.py] for particle in self.particles]
        #for x, y in zip(self.init_posX, self.init_posY):
        plt.scatter(x,y, c='b')
        plt.scatter(self.solnX, self.solnY, c='r', s=100)
        plt.legend(['particle', 'solution'])
        if show_it:
            plt.show()

    def save_report(self):
        if os.path.isfile(self.report_file_name):
            print('opening :{}'.format(self.report_file_name))
            report = pd.read_csv(self.report_file_name)
            report['y_loss'] = (self.ylosses + report['y_loss'])/2
            report['x_loss'] = (self.xlosses + report['x_loss'])/2
            report['loss'] = (self.losses + report['loss'])/2
            report['runs'] = list([report['runs'][0]+1]*len(self.losses))
            report.to_csv(self.report_file_name, index=False)

            reportCon = pd.read_csv(self.report_file_name_con)
            reportCon['runs'] = list([report['runs'][0]+1]*len(self.losses))
            reportCon['% converge'] = (self.percentage_converge + reportCon['% converge'] )/2
            reportCon.to_csv(self.report_file_name_con, index=False)
        else:
            report = pd.DataFrame()
            report['y_loss'] = self.ylosses
            report['x_loss'] = self.xlosses
            report['loss'] = self.losses
            report['runs'] = list([1]*len(self.losses))
            report.to_csv(self.report_file_name, index=False)

            reportCon = pd.DataFrame()
            reportCon['runs'] = list([1]*len(self.losses))
            reportCon['% converge'] = self.percentage_converge
            reportCon.to_csv(self.report_file_name_con, index=False)

    def plot_report_files(self, show_it=False, range=None):
        loss_data = pd.read_csv(self.report_file_name)
        if range is None:
            rng = len(loss_data['loss'].values.tolist())
        else:
            rng = min(range, len(loss_data['loss'].values.tolist()))
        self.losses = loss_data['loss'].values.tolist()[:rng]
        self.xlosses = loss_data['x_loss'].values.tolist()[:rng]
        self.ylosses = loss_data['y_loss'].values.tolist()[:rng]
        self.plot_average_loss()
        self.plot_xyaverage_loss()
        convergence_data = pd.read_csv(self.report_file_name_con)
        self.percentage_converge = convergence_data['% converge'].values.tolist()[:rng]
        self.percentage_plot()
        if show_it:
            plt.show()









