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
    if (receivedClockws == "anti-clockwise" and angle_new > angle_old) or (receivedClockws == "clockwise" and angle_new > angle_old):
        x = 1 * math.cos(angle_new + 0.5 * math.pi)
        y = 1 * math.sin(angle_new + 0.5 * math.pi)
        real_angle = angle_new
        print("up case")
        # print("up case: ({},{})".format(x, y))

    # down case
    elif (receivedClockws == "anti-clockwise" and angle_new < angle_old) or (receivedClockws == "clockwise" and angle_new > angle_old):
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

# 将向量标准化
def norm_vec(vec):
    vec_norm = np.linalg.norm(vec)
    vec_unit = vec / vec_norm
    return vec_unit

def ProcessAngle( angle_new):
    if angle_new < 0:
        angle_new = (-1) * angle_new
    angle_new -= 1.56

    return angle_new


class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.last_angle = 0
        self.time_step = 240  # ms
        self.time_experiment = 240  # s

        # Rewards
        self.CROSS_REWARD = 0.3
        self.GOAL_REWARD = 0.5
        self.ZONE_REWARD = 0.0001

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

        # zone
        # self.zone_index = 0
        self.temp_angle = 0
        self.temp_vector = 0
        self.zone_count = 0
        self.zone3_count = 0
        self.zone_fit = 0
        self.zone3_fit = 0

        self.potentialzone_left = [-0.23, 0.11, 0.18, 0.3 - 0.03]   # reward zone when turning left
        self.potentialzone_right = [0.35, 0.11, 0.18, 0.26 - 0.03]  # reward zone when turning right
        self.potentialzone_vertical = [-0.71, -0.38]                # y_low, y_high
        # self.potentialzone_horizontal = []

        self.vec_zone1 = norm_vec(np.array([-0.34, 0.12]))
        self.vec_zone2 = norm_vec(np.array([0.24, 0.08]))
        self.vec_zone3 = norm_vec(np.array([0, 1]))
        # self.vec_zone4

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
        string_message = string_message.encode("utf-8")
        # print("Supervisor send:", string_message)
        # self.emitter.send(string_message)

        # 只有到达交叉点后才报告
        if self.crossflag:
            x = "1"
        else:
            x = "0"
        data = x
        data = data.encode("utf-8")
        string_message = string_message + data
        # string_message = string_message.encode("utf-8")
        # print("Supervisor send:", string_message)
        self.emitter.send(string_message)

    def zone_product(self, cur_pos):
        ###### VERY IMPORTANT STEP ######
        self.temp_angle = self.rot_field.getSFRotation()[3]

        # Step 1: make y-axis to x-axis
        self.temp_angle = ProcessAngle(self.temp_angle)
        # print(self.temp_angle)

        # Step 2:
        self.temp_vector = correctVector(self.last_angle, self.temp_angle, self.receivedClockws)
        self.temp_vector = norm_vec(self.temp_vector)
        # print(self.temp_vector)

        zone3_fit = 0
        if self.potentialzone_vertical[0] <= cur_pos[1] <= self.potentialzone_right[1]:
            zone3_fit = (self.vec_zone3 @ self.temp_vector - 1) / 20
            # print("Zone3 {}".format(zone3_fit))
            self.zone3_count += 1

        ###
        zone_fit = 0

        if self.crossflag and self.light_flag.getSFBool() and self.potentialzone_left[2] <= cur_pos[1] <= self.potentialzone_left[3]:
            zone_fit = 0.5 * self.vec_zone1 @ self.temp_vector      # origin: 2
            print("Zone1 {}".format(zone_fit))
            self.zone_count += 1

        elif self.crossflag and not self.light_flag.getSFBool() and self.potentialzone_right[2] <= cur_pos[1] <= self.potentialzone_right[3]:
            zone_fit = 0.5 * self.vec_zone2 @ self.temp_vector      # origin: 2
            print("Zone2 {}".format(zone_fit))
            self.zone_count += 1

        else:
            pass

        # if a wrong direction, assign negative gain
        # if zone_fit < -0.1:
        #     zone_fit = -1.5
        self.last_angle = self.temp_angle
        return zone_fit, zone3_fit

    def run_seconds(self, seconds):
        # print(self.left_motor.getSFVec3f())

        # print("Run Simulation")
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0
        self.zone_fit = 0
        self.zone3_fit = 0
        lock_1 = True
        self.crossflag = False

        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if stop == iterations:
                break
            # print(self.supervisor.step(self.time_step))
            cur_pos = self.trans_field.getSFVec3f()

            # Check robot in Cross Zone [0.05~0.18;0.12~0.25]
            if self.crosszone[0] < self.trans_field.getSFVec3f()[0] < self.crosszone[1] \
                    and self.crosszone[2] < self.trans_field.getSFVec3f()[1] < self.crosszone[3] and not self.crossflag:
                print("####### Cross Zone ########")
                self.crossflag = True

            # Calculate the light+turning reward
            cur_fit, cur3_fit = self.zone_product(cur_pos)
            self.zone_fit += cur_fit
            self.zone3_fit += cur3_fit

            iterations = iterations + 1

        # calculate zone 1 or zone 2 fit
        if self.zone_count != 0:
            # print(self.zone_count)
            self.zone_fit = self.zone_fit / self.zone_count
            self.zone_count = 0       # Repeat from zero in next loop

        # calculate zone 1 or zone 2 fit
        if self.zone3_count != 0:
            # print(self.zone3_count)
            self.zone3_fit = self.zone3_fit / self.zone3_count
            self.zone3_count = 0  # Repeat from zero in next loop

    def evaluate_genotype(self, genotype, generation):
        # Here you can choose how many times the current individual will interact with both environments
        # At each interaction loop, one trial on each environment will be performed
        numberofInteractionLoops = 3
        currentInteraction = 0
        fitnessPerTrial = []
        while currentInteraction < numberofInteractionLoops:
            #######################################
            # TRIAL: TURN RIGHT
            #######################################
            # Send genotype to robot for evaluation
            self.emitterData = str(genotype)

            global real_angle
            real_angle = 0

            # Reset robot position and physics
            INITIAL_TRANS = [-0.686, -0.66, 0]
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [0, 0, 1, 1.63]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()

            # Spotlight turns OFF
            self.light_flag.setSFBool(False)
            print("light flag = {}".format(self.light_flag.getSFBool()))
            self.light_node.resetPhysics()

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
            if self.zone_fit != 0:
                fitness = float(fitness) + self.zone_fit
                print("###### Turning Reward: {} ###### ".format(self.zone_fit))

            # zone 3
            if self.zone3_fit != 0:
                fitness = float(fitness) + self.zone3_fit
                print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))

            #### Goal Reward
            final_position = self.trans_field.getSFVec3f()
            # print("Final Position: {} ".format(final_position))
            if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[2] < -0.14:
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
            INITIAL_TRANS = [-0.686, -0.66, 0]
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [0, 0, 1, 1.63]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()

            # Spotlight turns ON
            self.light_flag.setSFBool(True)
            print("light flag = {}".format(self.light_flag.getSFBool()))
            self.light_node.resetPhysics()

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
            if self.zone_fit != 0:
                fitness = float(fitness) + self.zone_fit
                print("###### Turning Reward: {} ###### ".format(self.zone_fit))

            # zone 3
            if self.zone3_fit != 0:
                fitness = float(fitness) + self.zone3_fit
                print("###### Going Down Debuff: {} ###### ".format(self.zone3_fit))

            #### Goal Reward
            final_position = self.trans_field.getSFVec3f()
            # print("Final Position: {} ".format(final_position))
            if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[2] < -0.14:
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
            np.save("Best.npy", best[0])
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
        self.light_node.resetPhysics()

        # Evaluation genotype
        self.run_seconds(self.time_experiment)

        # Measure fitness
        fitness = self.receivedFitness

        #### Cross Reward
        if self.crossflag:
            fitness = float(fitness) + self.CROSS_REWARD

        #### Turning Zone Reward:
        fitness = float(fitness) + self.zone_fit

        #### Goal Reward
        final_position = self.trans_field.getSFVec3f()
        # print("Final Position: {} ".format(final_position))
        if 0.33 < final_position[0] < 0.38 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[
            2] < -0.14:
            fitness = float(fitness) + self.GOAL_REWARD

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
        self.light_node.resetPhysics()

        # Evaluation genotype
        self.run_seconds(self.time_experiment)

        # Measure fitness
        fitness = self.receivedFitness

        ################
        #### REWARD ####
        ################
        if self.crossflag:  #### Cross Reward
            fitness = float(fitness) + self.CROSS_REWARD
        fitness = float(fitness) + self.zone_fit / self.zone_count  #### Zone Reward
        final_position = self.trans_field.getSFVec3f()  #### Goal Reward
        # print(final_position)
        if -0.38 < final_position[0] < -0.33 and -0.01 < final_position[1] < 0.01 and -0.18 < final_position[
            2] < -0.14:
            fitness = float(fitness) + 0.5

        print("Fitness: {}".format(fitness))

    def draw_scaled_line(self, generation, y1, y2):
        # the scale of the fitness plot
        XSCALE = int(self.width / self.num_generations)
        YSCALE = 100
        self.display.drawLine((generation - 1) * XSCALE, self.height - int(y1 * YSCALE), generation * XSCALE,
                              self.height - int(y2 * YSCALE))

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