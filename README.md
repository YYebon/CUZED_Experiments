
# CUZED_Experiments
### Experiments about paper CUZED
## CUZED works with this flows 

![image](https://github.com/YYebon/CUZED_Experiments/assets/148024646/8db4c4f8-eb29-41cc-b7e8-0bb27217693c)

### The codes are running with Pytorch environment
### Faults are injected directly with random value (change random value)

Example (layer[v[r][0]][v[r][1]] is specific fully connected layers weight )

r = random.randrange(0,num)
temp = layer[v[r][0]][v[r][1]].clone() #copy the default value
layer[v[r][0]][v[r][1]] = 10**random.randrange(3, 10)#*random.randrange(10,20) insert random weight value in random neruon which located in specific layer(fc1)
### The experiment was applied to the following networks and datasets
CNN
Alexnet
VGG-16

MNIST
CIFAR-10
