"""supervisorGA-ER controller."""
import math

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import ga, os, sys, struct

real_angle = 0


# tell
def correctVector(angle_old, angle_new, receivedClockws):
    global real_angle

    # left: 1.52, right: -1.52
    # up case
    # print(receivedClockws)
    print(angle_new)
    if (receivedClockws == "anti-clockwise" and angle_new > angle_old) or (
            receivedClockws == "clockwise" and angle_new > angle_old):
        x = 1 * math.cos(angle_new + 0.5 * math.pi)
        y = 1 * math.sin(angle_new + 0.5 * math.pi)
        real_angle = angle_new
        print("up case")
        # print("up case: ({},{})".format(x, y))

    # down case
    elif (receivedClockws == "anti-clockwise" and angle_new < angle_old) or (
            receivedClockws == "clockwise" and angle_new > angle_old):
        x = 1 * math.cos(angle_new + 0.5 * math.pi)
        y = (-1) * math.sin(angle_new + 0.5 * math.pi)
        print("down case")
        # print("down case: ({},{})".format(x, y))
        real_angle = -angle_new

    else:
        x = 1 * math.cos(real_angle + 0.5 * math.pi)
        y = 1 * math.sin(real_angle + 0.5 * math.pi)
    # print(real_angle)
    return np.array([x, y])


# 获取向量并将向量标准化
def getVector(old_trans, cur_trans):
    vec = np.array([cur_trans[0] - old_trans[0], cur_trans[1] - old_trans[1]])
    return vec


def norm_vec(vec):
    vec_norm = np.linalg.norm(vec)
    vec_unit = vec / vec_norm
    return vec_unit


def ProcessAngle(angle_new):
    if angle_new < 0:
        angle_new = (-1) * angle_new
    angle_new -= 1.56

    return angle_new


class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.next_trial_flag = False
        self.cur_vector = None
        self.last_angle = 0
        self.time_step = 96  # ms
        self.time_experiment = 180  # s

        # Rewards
        self.CROSS_REWARD = 30 # 3
        self.GOAL_REWARD = 100 # 10
        self.ZONE_REWARD = 0.0001
        self.OBSTACLE_PASS_REWARD = 20       # 0.5
        self.OBSTACLE_PASS_REWARD_LOW = 13   # 0.3

        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        self.cur_trans = self.trans_field.getSFVec3f()
        self.old_trans = self.trans_field.getSFVec3f()

        # Get obstacles translation and rotation current parameters
        self.obstacle_1_node = self.supervisor.getFromDef("OBS_Box1")
        self.obstacle_1_trans_field = self.obstacle_1_node.getField("translation")
        self.obstacle_1_rot_field = self.obstacle_1_node.getField("rotation")
        self.obstacle_2_node = self.supervisor.getFromDef("OBS_Cyn1")
        self.obstacle_2_field = self.obstacle_2_node.getField("translation")
        self.obstacle_3_node = self.supervisor.getFromDef("OBS_Box2")
        self.obstacle_3_trans_field = self.obstacle_3_node.getField("translation")
        self.obstacle_3_rot_field = self.obstacle_3_node.getField("rotation")
        self.obstacle_4_node = self.supervisor.getFromDef("OBS_Cyn2")
        self.obstacle_4_field = self.obstacle_4_node.getField("translation")

        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)

        # Initialize the receiver and emitter data to null
        self.receivedData = ""
        self.receivedWeights = ""
        self.receivedFitness = ""
        self.emitterData = ""

        # cross
        # CrossRoad[0.05~0.18;0.12~0.25]
        self.crosszone = [0.05, 0.18, 0.12, 0.25]
        self.crossflag = False

        # Obstacle-Passing Reward Zone
        delta = 0.02
        self.obstacle_pass_zone_1 = [-0.29 - delta, 0.47 - delta, -0.29 + delta, 0.47 + delta]
        self.obstacle_pass_zone_2 = [0.48 - delta, 0.37 - delta, 0.48 + delta, 0.37 + delta]
        self.obstacle_flag = False

        # zone
        # self.zone_index = 0
        self.temp_angle = 0
        self.temp_vector = 0
        self.zone12_count = 0
        self.zone12_fit = 0
        self.zone3_count = 0
        self.zone3_fit = 0
        self.zone4_count = 0
        self.zone4_fit = 0

        self.potentialzone_left = [-0.20, 0.11, 0.18, 0.22]  # x1, x2, y1, y2
        self.potentialzone_right = [0.35, 0.11, 0.18, 0.21]  # reward zone when turning right
        self.potentialzone_vertical = [-0.71, -0.38]  # y_low, y_high
        self.potentialzone_horizontal = [-0.54, -0.008, -0.32, -0.07]  # x_left, x_right, y_bottom, y_top

        self.vec_zone1 = norm_vec(np.array([-3.4, 1.2]))  # -0.34, 0.12
        self.vec_zone2 = norm_vec(np.array([2.4, 0.8]))   # 0.24, 0.08
        self.vec_zone3 = norm_vec(np.array([0, 10]))      # [0, 10]
        self.vec_zone4 = norm_vec(np.array([10, 0]))      # [10, 0]

        ###########
        ### DEFINE here the 3 GA Parameters:
        self.num_generations = 200
        self.num_population = 15
        self.num_elite = 2

        # size of the genotype variable
        self.num_weights = 0

        # Creating the initial population
        self.population = []

        # All Genotypes
        self.genotypes = []

        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0, 0)
        self.display.drawText("Fitness (Average - Green)", 0, 10)

        # Get the light node from your world environment
        self.light_node = self.supervisor.getFromDef("Light")
        if self.light_node is None:
            sys.stderr.write("No DEF SpotLight node found in the current world file\n")
            sys.exit(1)
        self.light_flag = self.light_node.getField("on")

    real_angle = 0

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if (self.num_weights > 0):
            #  Size of the population and genotype
            pop_size = (self.num_population, self.num_weights)
            # Create the initial population with random weights
            self.population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while (self.receiver.getQueueLength() > 0):
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # Check Message
            if (typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)]
                self.num_weights = int(self.receivedWeights)
            elif (typeMessage == "fitness"):
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            elif (typeMessage == "clockws"):
                self.receivedClockws = self.receivedData[9:len(self.receivedData)]
                # print("received:" + str(self.receivedClockws))
            self.receiver.nextPacket()

    def handle_emitter(self):
        data = str(self.emitterData)
        data = "genotype: " + data
        string_message = str(data)
        # string_message = string_message.encode("utf-8")
        # print("Supervisor send:", string_message)
        # self.emitter.send(string_message)

        # crossflag
        if self.crossflag:
            x = "1"
        else:
            x = "0"
        data = x
        # data = data.encode("utf-8")
        string_message = string_message + data
        # string_message = string_message.encode("utf-8")

        # 只有到达交叉点后才报告
        if self.next_trial_flag:
            x = "1"
        else:
            x = "0"
        data = x
        # data = data.encode("utf-8")
        string_message = string_message + data

        string_message = string_message.encode("utf-8")
        # print("Supervisor send:", string_message)
        self.emitter.send(string_message)

    # Calculate the moving_up, light_turning reward
    # 1. new_trans - old_trans --> cur_vector
    # 2. cur_vector (dot_product) zone_vector --> reward unit
    # 3. accumulate reward units and compute the average
    def zone_product(self, cur_vec, cur_trans):
        # Light Turning Reward
        zone12_fit = 0

        # zone1: Turn Left
        if self.crossflag and self.light_flag.getSFBool() and self.potentialzone_left[2] <= cur_trans[1] <= \
                self.potentialzone_left[3]:
            zone12_fit = 0.4 * self.vec_zone1 @ cur_vec  # origin: 2
            # print("Zone1 {}".format(zone12_fit))
            self.zone12_count += 1

        # zone2: Turn Right
        elif self.crossflag and not self.light_flag.getSFBool() and self.potentialzone_right[2] <= cur_trans[1] <= \
                self.potentialzone_right[3]:
            zone12_fit = 0.4 * self.vec_zone2 @ cur_vec  # origin: 2
            # print("Zone2 {}".format(zone12_fit))
            self.zone12_count += 1
        else:
            pass

        # zone3: Move Vertical Up
        zone3_fit = 0
        if self.potentialzone_vertical[0] <= cur_trans[1] <= self.potentialzone_vertical[1]:
            zone3_fit = 0.2 * (self.vec_zone3 @ cur_vec - 1)
            # print("Zone3 {}".format(zone3_fit))
            self.zone3_count += 1

        # zone 4: Move Horizontal Right
        zone4_fit = 0
        if self.potentialzone_horizontal[0] <= cur_trans[0] <= self.potentialzone_horizontal[1] \
                and self.potentialzone_horizontal[2] <= cur_trans[1] <= self.potentialzone_horizontal[3]:
            zone4_fit = 0.2 * (self.vec_zone4 @ cur_vec - 1)
            # print("Zone4 {}".format(zone4_fit))
            self.zone4_count += 1

        return zone12_fit, zone3_fit, zone4_fit

    def run_seconds(self, seconds):

        # print("Run Simulation")
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0
        self.zone12_fit = 0
        self.zone3_fit = 0
        self.zone4_fit = 0
        lock_1 = True
        self.crossflag = False
        self.obstacle_flag = False

        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if stop == iterations:
                self.next_trial_flag = 1       # get ready for next trial
                break

            # if it's the first moment of a new trail, flag = 1
            # we need to set it back to 0 immediately
            if self.next_trial_flag == 1:
                self.next_trial_flag = 0

            # print(self.supervisor.step(self.time_step))
            self.cur_trans = self.trans_field.getSFVec3f()
            self.cur_vector = getVector(self.old_trans, self.cur_trans)
            self.cur_vector = norm_vec(self.cur_vector)

            # Check robot in Cross Zone [0.05~0.18;0.12~0.25]
            if self.crosszone[0] < self.cur_trans[0] < self.crosszone[1] \
                    and self.crosszone[2] < self.cur_trans[1] < self.crosszone[3] and not self.crossflag:
                print("####### Cross Zone ########")
                self.crossflag = True

            # Check Obstacle Avoidance
            if self.crossflag:  # and self.light_flag
                # left obstacle
                if self.obstacle_pass_zone_1[0] < self.cur_trans[0] < self.obstacle_pass_zone_1[2] \
                        and self.obstacle_pass_zone_1[1] < self.cur_trans[1] < self.obstacle_pass_zone_1[3]:
                    print("####### Successful Avoid Obstacle 1 ########")
                    self.obstacle_flag = True
                # right obstacle
                elif self.obstacle_pass_zone_2[0] < self.cur_trans[0] < self.obstacle_pass_zone_2[2] \
                        and self.obstacle_pass_zone_2[1] < self.cur_trans[1] < self.obstacle_pass_zone_2[3]:
                    print("####### Successful Avoid Obstacle 2 ########")
                    self.obstacle_flag = True


            # Calculate the light+turning reward
            cur12_fit, cur3_fit, cur4_fit = self.zone_product(self.cur_vector, self.cur_trans)
            self.zone12_fit += cur12_fit
            self.zone3_fit += cur3_fit
            self.zone4_fit += cur4_fit

            # next loop
            self.old_trans = self.cur_trans
            iterations = iterations + 1

        # calculate zone 1 or zone 2 fit
        if self.zone12_count != 0:
            # print(self.zone_count)
            self.zone12_fit = self.zone12_fit / self.zone12_count
            self.zone12_count = 0  # Repeat from zero in next loop

        # calculate zone 3 fit
        if self.zone3_count != 0:
            # print(self.zone3_count)
            self.zone3_fit = self.zone3_fit / self.zone3_count
            self.zone3_count = 0  # Repeat from zero in next loop

        # calculate zone 4 fit
        # print("Zone 4 {} ".format(self.zone4_count))
        if self.zone4_count != 0:
            self.zone4_fit = self.zone4_fit / self.zone4_count
            self.zone4_count = 0  # Repeat from zero in next loop

    def evaluate_genotype(self, genotype, generation):
        # Here you can choose how many times the current individual will interact with both environments
        # At each interaction loop, one trial on each environment will be performed
        numberofInteractionLoops = 3
        currentInteraction = 0
        fitnessPerTrial = []

        # reset obstacle position
        INIT_OBSTACLE_1_TRANS = [0.35, 0.26, 0.0496]  # Rectangle 1
        INIT_OBSTACLE_1_ROT = [-0.579, - 0.579, 0.574, 0]
        self.obstacle_1_trans_field.setSFVec3f(INIT_OBSTACLE_1_TRANS)
        self.obstacle_1_rot_field.setSFRotation(INIT_OBSTACLE_1_ROT)
        INIT_OBSTACLE_2 = [-0.23, 0.3, 0.05]  # Cylinder 1
        self.obstacle_2_field.setSFVec3f(INIT_OBSTACLE_2)
        INIT_OBSTACLE_3_TRANS = [-0.21, 0.69, 0.0496]  # Rectangle 2
        INIT_OBSTACLE_3_ROT = [0, 0, 1, 0]
        self.obstacle_3_trans_field.setSFVec3f(INIT_OBSTACLE_3_TRANS)
        self.obstacle_3_rot_field.setSFRotation(INIT_OBSTACLE_3_ROT)
        INIT_OBSTACLE_4 = [0.38, 0.71, 0.05]  # Cylinder 2
        self.obstacle_4_field.setSFVec3f(INIT_OBSTACLE_4)

        while currentInteraction < numberofInteractionLoops:
            #######################################
            # TRIAL: TURN RIGHT
            #######################################
            # Send genotype to robot for evaluation
            self.emitterData = str(genotype)

            global real_angle
            real_angle = 0


            # Reset robot position and physics
            # INITIAL_TRANS = [-0.686, -0.66, 0]
            INITIAL_TRANS = [0.1, 0.04, 0.049]   # for test
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [0, 0, 1, 1.63]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()

            # Spotlight turns OFF
            self.light_flag.setSFBool(False)
            # print("light flag = {}".format(self.light_flag.getSFBool()))
            self.light_node.resetPhysics()

            self.next_trial_flag = 1

            # Evaluation genotype
            self.run_seconds(self.time_experiment)

            # Measure fitness
            fitness = self.receivedFitness

            ############## REWARD ###############
            #### Cross Reward
            if self.crossflag:
                fitness = float(fitness) + self.CROSS_REWARD
                print("###### Crossroad Reward ###### ")

            #### Turning Zone Reward
            # zone 1 or 2
            if self.zone12_fit != 0:
                fitness = float(fitness) + self.zone12_fit
                print("###### Turning Reward: {} ###### ".format(self.zone12_fit))

            # zone 3
            if self.zone3_fit != 0:
                fitness = float(fitness) + self.zone3_fit
                print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))

            # zone 4
            if self.zone4_fit != 0:
                fitness = float(fitness) + self.zone4_fit
                print("###### Going Left Debuff: {} ###### ".format(self.zone4_fit))

            #### Obstacle Avoidance Reward (right case)
            if self.obstacle_flag:
                fitness = float(fitness) + self.OBSTACLE_PASS_REWARD

            #### Goal Reward
            final_position = self.trans_field.getSFVec3f()
            # print("Final Position: {} ".format(final_position))
            if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[
                2] < -0.14:
                fitness = float(fitness) + self.GOAL_REWARD
                print("###### Goal Reward ###### ")
            print("Fitness: {}".format(fitness))
            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)

            #######################################
            # TRIAL: TURN LEFT
            #######################################
            # Send genotype to robot for evaluation
            self.emitterData = str(genotype)

            real_angle = 0

            # Reset robot position and physics
            # INITIAL_TRANS = [-0.686, -0.66, 0]
            INITIAL_TRANS = [0.1, 0.04, 0.049]
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [0, 0, 1, 1.63]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()

            # Spotlight turns ON
            self.light_flag.setSFBool(True)
            # print("light flag = {}".format(self.light_flag.getSFBool()))
            self.light_node.resetPhysics()

            self.next_trial_flag = 1

            # Evaluation genotype
            self.run_seconds(self.time_experiment)

            # Measure fitness
            fitness = self.receivedFitness

            ############## REWARD ###############
            #### Cross Reward
            if self.crossflag:
                fitness = float(fitness) + self.CROSS_REWARD
                print("###### Crossroad Reward ###### ")

            #### Turning Zone Reward
            # zone 1 or 2
            if self.zone12_fit != 0:
                fitness = float(fitness) + self.zone12_fit
                print("###### Turning Reward: {} ###### ".format(self.zone12_fit))

            # zone 3
            if self.zone3_fit != 0:
                fitness = float(fitness) + self.zone3_fit
                print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))

            # zone 4
            if self.zone4_fit != 0:
                fitness = float(fitness) + self.zone4_fit
                print("###### Going Left Debuff: {} ###### ".format(self.zone4_fit))

            #### Obstacle Avoidance Reward (left case)
            if self.obstacle_flag:
                fitness = float(fitness) + self.OBSTACLE_PASS_REWARD

            #### Goal Reward
            final_position = self.trans_field.getSFVec3f()
            # print("Final Position: {} ".format(final_position))
            if self.crossflag and self.obstacle_flag:
                if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[
                    2] < -0.14:
                    fitness = float(fitness) + self.GOAL_REWARD
                    print("###### Goal Reward ###### ")
            print("Fitness: {}".format(fitness))

            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)

            # End
            currentInteraction += 1

        print(fitnessPerTrial)

        fitness = np.mean(fitnessPerTrial)
        current = (generation, genotype, fitness)
        self.genotypes.append(current)

        return fitness

    def run_optimization(self):
        # Wait until the number of weights is updated
        while (self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()

        print(">>>Starting Evolution using GA optimization ...\n")

        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []
            # Select each Genotype or Individual
            for population in range(self.num_population):
                # print("test")
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype, generation)
                # Save its fitness value
                current_population.append((genotype, float(fitness)))
                # print(current_population)

            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population)
            average = ga.getAverageGenotype(current_population)
            # np.save("Best.npy", best[0])
            np.save('Best_' + str(generation) + '.npy', best[0])
            self.plot_fitness(generation, best[1], average)

            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population, self.num_elite)

        # print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")

    def run_demo(self):
        # Read File
        genotype = np.load("Best.npy")

        # Send Genotype to controller
        self.emitterData = str(genotype)

        # Reset robot position and physics
        INITIAL_TRANS = [-0.686, -0.66, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 0, 1, 1.63]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()

        # Spotlight turns OFF
        self.light_flag.setSFBool(False)
        # print("light flag = {}".format(self.light_flag.getSFBool()))
        self.light_node.resetPhysics()

        self.next_trial_flag = 1
        # Evaluation genotype
        self.run_seconds(self.time_experiment)

        # Measure fitness
        fitness = self.receivedFitness

        ############## REWARD ###############
        #### Cross Reward
        if self.crossflag:
            fitness = float(fitness) + self.CROSS_REWARD
            print("###### Crossroad Reward ###### ")

        #### Turning Zone Reward:
        # zone 1 or 2
        if self.zone12_fit != 0:
            fitness = float(fitness) + self.zone12_fit
            print("###### Turning Reward: {} ###### ".format(self.zone12_fit))
        # zone 3
        if self.zone3_fit != 0:
            fitness = float(fitness) + self.zone3_fit
            print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))
        # zone 4
        if self.zone4_fit != 0:
            fitness = float(fitness) + self.zone4_fit
            print("###### Going Left Debuff: {} ###### ".format(self.zone4_fit))

        #### Obstacle Avoidance Reward (right case)
        if self.obstacle_flag:
            fitness = float(fitness) + self.OBSTACLE_PASS_REWARD

        #### Goal Reward
        final_position = self.trans_field.getSFVec3f()
        # print("Final Position: {} ".format(final_position))
        if self.crossflag and self.obstacle_flag:
            if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01:
                fitness = float(fitness) + self.GOAL_REWARD
                print("###### Goal Reward ###### ")

        print("Fitness: {}".format(fitness))

        #######################################
        # TRIAL: TURN LEFT
        #######################################
        # Send genotype to robot for evaluation
        self.emitterData = str(genotype)

        # Reset robot position and physics
        INITIAL_TRANS = [-0.686, -0.66, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 0, 1, 1.63]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()

        # Spotlight turns ON
        self.light_flag.setSFBool(True)
        # print("light flag = {}".format(self.light_flag.getSFBool()))
        self.light_node.resetPhysics()

        self.next_trial_flag = 1
        # Evaluation genotype
        self.run_seconds(self.time_experiment)

        # Measure fitness
        fitness = self.receivedFitness

        ############## REWARD ###############
        #### Cross Reward
        if self.crossflag:
            fitness = float(fitness) + self.CROSS_REWARD
            print("###### Crossroad Reward ###### ")

        #### Turning Zone Reward
        # zone 1 or 2
        if self.zone12_fit != 0:
            fitness = float(fitness) + self.zone12_fit
            print("###### Turning Reward: {} ###### ".format(self.zone12_fit))

        # zone 3
        if self.zone3_fit != 0:
            fitness = float(fitness) + self.zone3_fit
            print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))

        # zone 4
        if self.zone4_fit != 0:
            fitness = float(fitness) + self.zone4_fit
            print("###### Going Left Debuff: {} ###### ".format(self.zone4_fit))

        #### Obstacle Avoidance Reward (left case)
        if self.obstacle_flag:
            fitness = float(fitness) + self.OBSTACLE_PASS_REWARD

        #### Goal Reward
        final_position = self.trans_field.getSFVec3f()
        # print("Final Position: {} ".format(final_position))
        if self.crossflag and self.obstacle_flag:
            if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01:
                fitness = float(fitness) + self.GOAL_REWARD
                print("###### Goal Reward ###### ")
        print("Fitness: {}".format(fitness))


    def draw_scaled_line(self, generation, y1, y2):
        # the scale of the fitness plot
        XSCALE = int(self.width / self.num_generations)
        YSCALE = 100
        self.display.drawLine((generation - 1) * XSCALE, self.height - int(y1 * YSCALE), generation * XSCALE, self.height - int(y2 * YSCALE))

    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000)  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness)

            self.display.setColor(0x00ff00)  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness)

        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness


if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module
    gaModel = SupervisorGA()

    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)
    # Interface
    print("***************************************************************************************************")
    print("To start the simulation please click anywhere in the SIMULATION WINDOW(3D Window) and press either:")
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    print("***************************************************************************************************")

    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if (resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            # print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        elif (resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            # print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")

    # def inZone(self, x, y):
    #     if self.potentialzone_1[0] < x < self.potentialzone_1[1] and \
    #             self.potentialzone_1[2] < y < self.potentialzone_1[3]:
    #         index = 1
    #
    #     elif self.potentialzone_2[0] < x < self.potentialzone_2[1] and \
    #             self.potentialzone_2[2] < y < self.potentialzone_2[3]:
    #         index = 2
    #     return index
