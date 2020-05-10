using System;
using System.Collections.Generic;

using Nomad.Matrix;
using Vortex.Cost.Kernels;
using Vortex.Cost.Utility;

using Vortex.Layer.Kernels;
using Vortex.Layer.Utility;

using Vortex.Optimizer.Kernels;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;

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

        public Network(BaseCost cost, BaseOptimizer optimizer, int batchSize = 1)
        {
            IsLocked = false;
            Layers = new List<BaseLayerKernel>();
            BatchSize = batchSize;
            _currentBatch = 0;

            // Cost Function Setup
            CostFunction = cost.Type() switch
            {
                ECostType.CrossEntropyCost => new CrossEntropyCostKernel(),
                ECostType.ExponentionalCost => new ExponentialCostKernel((ExponentialCost)cost),
                ECostType.GeneralizedKullbackLeiblerDivergence => new GeneralizedKullbackLeiblerDivergenceKernel(),
                ECostType.HellingerDistance => new HellingerDistanceKernel(),
                ECostType.ItakuraSaitoDistance => new ItakuraSaitoDistanceKernel(),
                ECostType.KullbackLeiblerDivergence => new KullbackLeiblerDivergenceKernel(),
                ECostType.QuadraticCost => new QuadraticCostKernel(),
                _ => throw new ArgumentException("Cost Type Invalid.")
            };

            // Optimizer Function Setup
            OptimizerFunction = optimizer.Type() switch
            {
                EOptimizerType.AdaDelta => new AdaDeltaKernel((AdaDelta)optimizer),
                EOptimizerType.AdaGrad => new AdaGradKernel((AdaGrad)optimizer),
                EOptimizerType.Adam => new AdamKernel((Adam)optimizer),
                EOptimizerType.Adamax => new AdamaxKernel((Adamax)optimizer),
                EOptimizerType.GradientDescent => new GradientDescentKernel((GradientDescent)optimizer),
                EOptimizerType.Momentum => new MomentumKernel((Momentum)optimizer),
                EOptimizerType.Nadam => new NadamKernel((Nadam)optimizer),
                EOptimizerType.NesterovMomentum => new NesterovMomentumKernel((NesterovMomentum)optimizer),
                EOptimizerType.RmsProp => new RmsPropKernel((RmsProp)optimizer),
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
