



# CUZED_Experiments
### Experiments about paper CUZED

[CUZED: Error Correction Using Zero-Weight
upon Low-Cost Error Detection on Fully Connected Layers
.pdf](https://github.com/YYebon/CUZED_Experiments/files/13184234/CUZED_FINAL.pdf)

### CUZED works with this flows 
#### Based on any error-check method on FC layers
(used Sanity-Check in this paper)


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

