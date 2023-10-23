



# CUZED_Experiments
### Experiments about paper CUZED
##### https://koreaai.org/sites/koreaai2023/media?key=site/koreaai2023/abs/F-1-1.pdf


### CUZED works with this flows 
#### Based on Sanity-Check’s spatial checksum on FC layers
![image](https://github.com/YYebon/CUZED_Experiments/assets/148024646/8f697b0a-7da3-4d7d-a817-756b48e7b0ee)

First, it adds one more row in the weight matrix and sets the additional column values to the additive inverse of the summation of all values in the same column. 

Similarly, one additional bias value is required and should equal the additive inverse of the sum of all other bias values. Interestingly, these modifications are equivalent to adding an extra neuron in the FC layer named the sanity neuron. 

Next, summing the all-neuron’s outputs and performing the check operation by adding the sanity neuron’s value with the summing outputs. This sequence is on the check neuron, and its result should be zero in the absence of errors.

#### Error Correction by eliminating a fulty neuron.
The proposed approach involves the following steps for error detection and correction: 
1) Utilize the Sanity-Check method to identify the layer where an error has occurred. Note that other effective detection schemes can be used if they exist. 2) If the accuracy drops by more than 5% from the training dataset, employ the CUZED technique. 
3) Remove the top 100 weight neurons in descending order, one by one, by setting their weights to zero. 
4) After each elimination, evaluate the accuracies of the computations. 
5) Repeat the process until the highest accuracy neuron state is achieved, ensuring that the output is obtained successfully.
![image](https://github.com/YYebon/CUZED_Experiments/assets/148024646/8db4c4f8-eb29-41cc-b7e8-0bb27217693c)
#### The codes are running with Pytorch environment

#### Faults are injected directly with random value (change random value)
Example : 
(layer[v[r][0]][v[r][1]] is specific fully connected layers weight )
```python
r = random.randrange(0,num)
temp = layer[v[r][0]][v[r][1]].clone() #copy the default value
layer[v[r][0]][v[r][1]] = 10**random.randrange(3, 10)#*random.randrange(10,20) insert random weight value in random neruon which located in specific layer(fc1)
```
### The experiment was applied to the following networks and datasets
CNN
Alexnet
VGG-16
	
MNIST
CIFAR-10

#### Experiment results
![image](https://github.com/YYebon/CUZED_Experiments/assets/148024646/170d0673-7268-4954-9bb1-ecbea425fed7)
evaluate CUZED on popular DNN models with two datasets. Our evaluation shows that CUZED can recover the reduced accuracy with a single large error, even when it is reduced to 12.4% when the original was 99%. We have even found that it gets better than the original accuracy. We have concluded that, through the experiment, CUZED performed efficiently in both error correction and accuracy enhancement.

