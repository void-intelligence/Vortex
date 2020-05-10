# Vortex: Deep Learning Playground [![Void-Intelligence](https://circleci.com/gh/void-intelligence/Vortex.svg?style=shield)](https://app.circleci.com/pipelines/github/void-intelligence/Vortex)

<p align="center">
  <img src="https://github.com/void-intelligence/Vortex/blob/master/resources/Vortex-Logo.png" alt="Vortex Logo">
</p>

Welcome To Vortex, The Void-Intelligence Artificial Cortex library for Neural Network and Deep Learning development. 

Vortex is built with the aim of allowing Deep-Learning researchers to test, and see the result of their new formulas in all aspects of the data-flow pipeline, without the need to read the source code. Researchers can write their custom functions / formulas for the currently developed (and planned) features of the library:

- Activation Functions
- Cost Functions
- Layers
- Optimizer Functions
- Regularization Methods
- Weight Initializers
- Normalizer Functions
- Genetic Mutation Algorithms
- Random Distribution Generators
- And many more

The architecture of this library is built in a way where you can easily create your functions and test out your ideas with absolute ease.

The entire architecture is built as a pair of ```Kernel``` classes and ```Utility``` classes, Kernel classes hold the main functionality of the component of the library while the utility allows object oriented flexibility across Vortex, so in short, you've got both functional and object oriented in the same pack!

## Vortex Quickstart

Let us write an example project that will learn an XOR table of 3 inputs, with 1 output predicting the final result.

First we need to import our namespaces:

```C#
// For the console
using System;

// Our Matrix Library
using Nomad.Matrix;

// Our Main Network class
using Vortex.Network;

// We need access to the Layers since we'll be creating them
using Vortex.Layer.Kernels;

// Activation Kernels
using Vortex.Activation.Kernels;

// Regularization Kernels
using Vortex.Regularization.Kernels;

// Loss / Cost function Kernels
using Vortex.Cost.Kernels.Legacy;

// Optimizer Functions
using Vortex.Optimizer.Kernels;

// Weight Initializer Kernels
using Vortex.Initializer.Kernels;
```

Now that these are done, let us create our first Network within the main function, the architecture we want to desing is going to be a simple dense network with 4 layrs all using tanh activation with weights initiated via normal distribution of numbers between ```-0.5``` and ```0.5```. The network will use ```QuadraticCost``` as the cost function and ```Adam``` as the optimizer function.

```C#
// Sequential Neural Network using QuadraticCost as the cost funciton and 
// Adam as the optimizer algorithm with a learning rate of 0.03
var net = new Sequential(new QuadraticCost(), new Adam(0.03));

// Fully Connected (Dense) layer with 3 inputs (our input layer) using Tanh activation function
net.CreateLayer(new FullyConnected(3, new Tanh()));

// Fully Connected (Dense) layer with 3 inputs using Tanh activation function
net.CreateLayer(new FullyConnected(3, new Tanh()));

// Fully Connected (Dense) layer with 3 inputs using Tanh activation function
net.CreateLayer(new FullyConnected(3, new Tanh()));

// Output layer with 1 inputs using Tanh activation function
net.CreateLayer(new Output(1, new Tanh()));
```

After we're done creating our Network, we need to initialize it's weights and biases, this task is super simple as we just need to call InitNetwork() on our ```Network``` object.

```C#
net.InitNetwork();
```

We now need to create our dataset, as was stated earlier, it will be an XOR table of 3.

First we need two matrix arrays holding our inputs and outputs:
(Don't worry, Tensors are coming)

```C#
var inputs = new List<Matrix>();
var outputs = new List<Matrix>();
```

Now let's create our data:

```C#
// 0 0 0    => 0
inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 0.0 } }));
outputs.Add(new Matrix(new double[,] { { 0.0 } }));

// 0 0 1    => 1
inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 1.0 } }));
outputs.Add(new Matrix(new double[,] { { 1.0 } }));

// 0 1 0    => 1
inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 0.0 } }));
outputs.Add(new Matrix(new double[,] { { 1.0 } }));

// 0 1 1    => 0
inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 1.0 } }));
outputs.Add(new Matrix(new double[,] { { 1.0 } }));

// 1 0 0    => 1
inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }));
outputs.Add(new Matrix(new double[,] { { 1.0 } }));

// 1 0 1    => 0
inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 1.0 } }));
outputs.Add(new Matrix(new double[,] { { 0.0 } }));

// 1 1 0    => 0
inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 0.0 } }));
outputs.Add(new Matrix(new double[,] { { 0.0 } }));

// 1 1 1    => 1
inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 1.0 } }));
outputs.Add(new Matrix(new double[,] { { 1.0 } }));
```

Now, it's time to train our network, let's do 8 tries on our dataset:

```C#
for (var i = 0; i < 8; i++)
{
    net.Train(inputs[i % 8], outputs[i % 8]);
}
```

And as simple as that, our network has learnt to predict the XOR table of 3 with 100% accuracy over 8 tries. (1 epoch)

```Output Correctness Tolerance is 0.1```

```C#
var correct = 0;
for (var i = 0; i < 10; i++)
{
    correct += Math.Abs(net.Forward(inputs[0])[0, 0]) < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[1])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[2])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[3])[0, 0]) < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[4])[0, 0]) - 1 < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[5])[0, 0]) < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[6])[0, 0]) < 0.1 ? 1 : 0;
    correct += Math.Abs(net.Forward(inputs[7])[0, 0]) - 1 < 0.1 ? 1 : 0;
}
var acc = correct / 80.0 * 100.0;
```

You can find the Gist of this example [here](https://gist.github.com/nirex0/77cdb951992a831ffc0343b0226b1513).

## Documentation

The full documentation will be ready as soon as the first release.
