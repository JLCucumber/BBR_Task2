"""epuck_python-ER controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import sys, struct, math
import numpy as np
import mlp as ntw


def clip_value(value, min_max):
    if (value > min_max):
        return min_max
    elif (value < -min_max):
        return -min_max
    return value



class Controller:
    def __init__(self, robot):
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 240  # ms
        self.max_speed = 1  # m/s

        # MLP Parameters and Variables

        ###########
        ### DEFINE below the architecture of your MLP network:
        ### Add the number of neurons for input layer, hidden layer and output layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.

        self.number_input_layer = 12    # Having added light sensor flag (11+1)
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = [7, 5]  # [?] OR
        self.number_output_layer = 2

        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)

        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1, len(self.number_neuros_per_layer)):
            if (n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n - 1] + 1) * self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
        self.original_left = 0
        self.original_right = 0
        self.clockwise = None

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # Enable Light Sensors
        self.lightOn = False
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.receivedData = ""
        self.receivedDataPrevious = ""
        self.receivedCross = False
        self.flagMessage = False


        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

    def findVelocity(self, left, right):
        # print("left{}, right{}".format(left, right))
        if left - right > 0.07:
            self.clockwise = "clockwise"
        elif left - right < -0.07:
            self.clockwise = "anti-clockwise"
        else:
            self.clockwise = "Straight"

    def check_for_new_genes(self):
        if self.flagMessage == True:
            # Split the list based on the number of layers of your network
            part = []
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    part.append((self.number_neuros_per_layer[n - 1] + 1) * (self.number_neuros_per_layer[n]))
                else:
                    part.append(self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n])

            # Set the weights of the network
            data = []
            weightsPart = []
            sum = 0
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart.append(self.receivedData[n - 1:part[n - 1]])
                elif (n == (len(self.number_neuros_per_layer) - 1)):
                    weightsPart.append(self.receivedData[sum:])
                else:
                    weightsPart.append(self.receivedData[sum:sum + part[n - 1]])
                sum += part[n - 1]
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1] + 1, self.number_neuros_per_layer[n]])
                else:
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1], self.number_neuros_per_layer[n]])
                data.append(weightsPart[n - 1])
            self.network.weights = data

            # Reset fitness list
            self.fitness_values = []

    def sense_compute_and_actuate(self):
        # MLP:
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.original_left = output[0]
        self.velocity_left = output[0] * 3
        if self.velocity_left >= self.max_speed:
            self.velocity_left = self.max_speed

        self.original_right = output[1]
        self.velocity_right = output[1] * 3
        if self.velocity_right >= self.max_speed:
            self.velocity_right = self.max_speed

        # print("left wheel {}, right wheel {}".format())
        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)


    def calculate_fitness(self):
        ########### Line (0~0.5)
        ### DEFINE the fitness function to follow the black line ground sensor: 0~1. When value close to 0, sensor detects the line### ground sensor: 0~1. When value close to 0, sensor detects the line
        temp_line = ((1-self.inputs[0]) + (1-self.inputs[1]) + (1-self.inputs[2])) / 3
        lineTrackFitness = temp_line - 0.25
        # print("lineTrackFitness value:  " + str(lineTrackFitness))  # "1 - self.input[n]"

        ########### Obstacle (0~1)
        proxi_max = 0
        for i in range(8):
            # print(self.proximity_sensors[i].getValue())
            temp = self.proximity_sensors[i].getValue()
            if temp > proxi_max:
                proxi_max = temp
        if proxi_max >= 1000: proxi_max = 1000          # 3. make obstacle to zero
        avoidCollisionFitness = proxi_max / 1000        # 1. 4000; 2. 2500
        # print("avoidCollisionFitness value:  " + str(1 - avoidCollisionFitness))

        ########### Forward (0~1)
        forwardFitness = (self.velocity_left + self.velocity_right) / 2
        # print("velocity: left {}, right{}".format(self.velocity_left, self.velocity_right))
        # print("forwardFitness value:  " + str(forwardFitness))

        ########### Spinning (0~1)
        spinningFitness = abs(self.velocity_left - self.velocity_right) / 2
        # print("spinningFitness value:  " + str(1 - math.sqrt(spinningFitness)))

        # 1. Special Case: Running straight into Obstacle, with a HIGH speed
        if avoidCollisionFitness < 0.1 and forwardFitness > 0.5 and spinningFitness > 0.5:
            forwardFitness = 0.1
            spinningFitness = 0.1

        ###########
        ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions
        ### 1. Line from "*" to "+"
        combinedFitness = lineTrackFitness + forwardFitness * (1 - math.sqrt(spinningFitness)) * (1 - avoidCollisionFitness)
        # print("combined Fitness value")

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send fitness:", string_message)
        self.emitter.send(string_message)

        self.findVelocity(self.original_left, self.original_right)
        data = str(self.clockwise)
        data = "clockws: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Sent:   " + str(self.clockwise))
        self.emitter.send(string_message)


    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while (self.receiver.getQueueLength() > 0):
                self.receivedData = self.receiver.getString()
                recived_data = self.receivedData
                # self.receivedData = self.receivedData[:-2]
                # print(self.receivedData)
                self.receivedData = self.receivedData[11:-2]
                # print(self.receivedData)

                # typeMessage = self.receivedData[0:9]
                # print(typeMessage)
                # if typeMessage == "genotype:":
                # Adjust the Data to our model
                # self.receivedData = self.receivedData[11:-2]
                # self.receivedData = self.receivedData.replace('[', '').replace(']', '')
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                # print(self.receivedData)

                # 收到交叉点到达信息（默认false）
                # elif typeMessage == "crossflg:":
                recived_data = self.receivedData[-1]
                self.receivedCross = recived_data
                if self.receivedCross == "1":
                    print(" CROSSROAD REACHED !!! ")
                    # print(self.receivedCross)

                self.receiver.nextPacket()

            # Is it a new Genotype?
            if (np.array_equal(self.receivedDataPrevious, self.receivedData) == False):
                self.flagMessage = True
            else:
                self.flagMessage = False

            self.receivedDataPrevious = self.receivedData
        else:
            # print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []

            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()

            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))

            ### Please adjust the ground sensors values to facilitate learning
            min_gs = 0
            max_gs = 1023

            if (left > max_gs): left = max_gs
            if (center > max_gs): center = max_gs
            if (right > max_gs): right = max_gs
            if (left < min_gs): left = min_gs
            if (center < min_gs): center = min_gs
            if (right < min_gs): right = min_gs

            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if (i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7):
                    temp = self.proximity_sensors[i].getValue()

                    ### Please adjust the distance sensors values to facilitate learning
                    min_ds = 0
                    max_ds = 4096   # 4096

                    if temp > max_ds: temp = max_ds
                    if temp < min_ds: temp = min_ds

                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # Read Light Sensors
            for i in range(8):
                temp = self.light_sensors[i].getValue()
                if temp <= 0.4:
                    # print("lights on")
                    self.lightOn = True
                    break
            self.inputs.append(int(self.lightOn))

            # GA Iteration
            # Verify if there is a new genotype to be used that was sent from Supervisor
            self.check_for_new_genes()
            # The robot's actuation (motor values) based on the output of the MLP
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()

            # End of the iteration

# Enter here exit cleanup code.
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()

# # create the Robot instance.
# robot = Robot()
#
# # get the time step of the current world.
# timestep = int(robot.getBasicTimeStep())
#
# # You should insert a getDevice-like function in order to get the
# # instance of a device of the robot. Something like:
# #  motor = robot.getDevice('motorname')
# #  ds = robot.getDevice('dsname')
# #  ds.enable(timestep)
#
# # Main loop:
# # - perform simulation steps until Webots is stopping the controller
# while robot.step(timestep) != -1:
#     # Read the sensors:
#     # Enter here functions to read sensor data, like:
#     #  val = ds.getValue()
#
#     # Process sensor data here.
#
#     # Enter here functions to send actuator commands, like:
#     #  motor.setPosition(10.0)
#     pass
