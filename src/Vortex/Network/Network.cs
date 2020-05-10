using System;
using System.Collections.Generic;

using Nomad.Matrix;
using Nomad.Utility;

using Vortex.Cost.Kernels;
using Vortex.Cost.Utility;

using Vortex.Layer.Kernels;
using Vortex.Layer.Utility;

using Vortex.Optimizer.Kernels;
using Vortex.Optimizer.Utility;

using Vortex.Initializer.Kernels;
using Vortex.Initializer.Utility;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Mutation.Utility;

namespace Vortex.Network
{
    public class Network
    {
        public float LastError { get; private set; }
        public Matrix Y { get; private set; }
        public List<BaseLayerKernel> Layers { get; }
        public bool IsLocked { get; private set; }
        public BaseOptimizerKernel OptimizerFunction { get; }
        public BaseCostKernel CostFunction { get; }
        public int BatchSize { get; set; }

        public Network(BaseCost costSettings, BaseOptimizer optimizerSettings, int batchSize = 1)
        {
            IsLocked = false;
            Layers = new List<BaseLayerKernel>();
            BatchSize = batchSize;
            _currentBatch = 0;

            // Cost Function Setup
            CostFunction = costSettings.Type() switch
            {
                ECostType.CrossEntropyCost => new CrossEntropyCostKernel((CrossEntropyCost)costSettings),
                ECostType.ExponentionalCost => new ExponentialCostKernel((ExponentionalCost)costSettings),
                ECostType.GeneralizedKullbackLeiblerDivergence => new GeneralizedKullbackLeiblerDivergenceKernel((GeneralizedKullbackLeiblerDivergence)costSettings),
                ECostType.HellingerDistance => new HellingerDistanceKernel((HellingerDistance)costSettings),
                ECostType.ItakuraSaitoDistance => new ItakuraSaitoDistanceKernel((ItakuraSaitoDistance)costSettings),
                ECostType.KullbackLeiblerDivergence => new KullbackLeiblerDivergenceKernel((KullbackLeiblerDivergence)costSettings),
                ECostType.QuadraticCost => new QuadraticCostKernel((QuadraticCost)costSettings),
                _ => throw new ArgumentException("Cost Type Invalid.")
            };

            // Optimizer Function Setup
            OptimizerFunction = optimizerSettings.Type() switch
            {
                EOptimizerType.AdaDelta => new AdaDeltaKernel((AdaDelta)optimizerSettings),
                EOptimizerType.AdaGrad => new AdaGradKernel((AdaGrad)optimizerSettings),
                EOptimizerType.Adam => new AdamKernel((Adam)optimizerSettings),
                EOptimizerType.Adamax => new AdamaxKernel((Adamax)optimizerSettings),
                EOptimizerType.GradientDescent => new GradientDescentKernel((GradientDescent)optimizerSettings),
                EOptimizerType.Momentum => new MomentumKernel((Momentum)optimizerSettings),
                EOptimizerType.Nadam => new NadamKernel((Nadam)optimizerSettings),
                EOptimizerType.NesterovMomentum => new NesterovMomentumKernel((NesterovMomentum)optimizerSettings),
                EOptimizerType.RmsProp => new RmsPropKernel((RmsProp)optimizerSettings),
                _ => throw new ArgumentException("Optimizer Type Invalid.")
            };
        }

        public void CreateLayer(BaseLayer layer)
        {
            if (IsLocked) throw new InvalidOperationException("Network is Locked.");

            // Layer Setup
            BaseLayerKernel layerKernel = layer.Type() switch
            {
                ELayerType.FullyConnected => new FullyConnectedKernel((FullyConnected)layer),
                ELayerType.Dropout => new FullyConnectedKernel((Dropout)layer),
                ELayerType.Output => new FullyConnectedKernel((Output)layer),
                _ => throw new ArgumentException("Invalid Layer Type.")
            };

            layerKernel.OptimizerFunction = OptimizerFunction;
            Layers.Add(layerKernel);
        }

        public void InitNetwork()
        {
            if (IsLocked) throw new InvalidOperationException("Network is Locked.");
            IsLocked = true;

            var result = new ResultKernel(new Result(Layers[^1].NeuronCount));
            Layers.Add(result);

            // Initialize All Layers, Their Ws and Bs
            for (var i = 0; i < Layers.Count - 1; i++)
            {
                // Weights
                Layers[i].Params["W"] = new Matrix(Layers[i + 1].NeuronCount, Layers[i].NeuronCount);
                
                if (i == 0 && Layers[i].Initializer.Type() == EInitializerType.Auto)
                {
                    Layers[i].Initializer.Scale *= Math.Sqrt(2.0 / (Layers[i].NeuronCount));
                    Layers[i].Params["W"] = Layers[i].Initializer.Initialize(Layers[i].Params["W"]);
                }
                else if (i != 0 && Layers[i].Initializer.Type() == EInitializerType.Auto)
                {
                    Layers[i].Initializer.Scale *= Math.Sqrt(2.0 / (Layers[i - 1].NeuronCount * Layers[i].NeuronCount));
                    Layers[i].Params["W"] = Layers[i].Initializer.Initialize(Layers[i].Params["W"]);
                }
                else
                {
                    Layers[i].Params["W"] = Layers[i].Initializer.Initialize(Layers[i].Params["W"]);
                }

                // Biases
                Layers[i].Params["B"] = new Matrix(Layers[i + 1].NeuronCount, 1);
                Layers[i].Params["B"].InFill(0);
            }
        }

        private int _currentBatch;

        private float _regularizationSum;
        
        private Matrix _actual;

        public Matrix Forward(Matrix input)
        {
            _regularizationSum = 0.0f;

            var yHat = input;
            foreach (var t in Layers)
            {
                yHat = t.Forward(yHat);
                _regularizationSum += t.RegularizationValue;
            }

            // Save data
            _actual = yHat;
            return yHat;
        }

        private void Backward(Matrix expected)
        {
            Y = expected;

            var da = CostFunction.Backward(_actual, expected);
            da *= LastError;

            // Calculate da for all layers
            for (var i = Layers.Count - 2; i >= 0; i--) da = Layers[i].Backward(da);

            // Optimize
            foreach (var t in Layers) t.Optimize();
        }

        public void Train(Matrix input, Matrix expected)
        {
            if (_currentBatch < BatchSize)
            {
                Forward(input);
                LastError += (float)CostFunction.Forward(_actual, expected);

                if (_currentBatch == 0)
                {
                    LastError += _regularizationSum;
                }
                _currentBatch++;
            }
            else if(_currentBatch == BatchSize)
            {
                Backward(expected);
                LastError = 0;
                _currentBatch = 0;
                _regularizationSum = 0;
            }

        }
    }
}
