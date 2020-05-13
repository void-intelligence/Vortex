// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;

using Nomad.Matrix;
using Vortex.Cost.Utility;

using Vortex.Layer.Kernels;
using Vortex.Layer.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;
using Vortex.Optimizer.Kernels;

namespace Vortex.Network
{
    public class Sequential
    {
        public double BatchError { get; set; }
        public double LastError { get; private set; }
        public Matrix Y { get; private set; }
        public bool IsLocked { get; private set; }
        public int BatchSize { get; set; }

        public List<BaseLayer> Layers { get; }
        public IOptimizer OptimizerFunction { get; }
        public ICost CostFunction { get; }

        public Sequential(ICost cost, IOptimizer optimizer, int batchSize = 1)
        {
            // If Optimizer is set to default use Adam
            OptimizerFunction = optimizer.Type() == EOptimizerType.Default ? new Adam(((BaseOptimizer)optimizer).Alpha, ((BaseOptimizer)optimizer).Decay) : optimizer;

            IsLocked = false;
            Layers = new List<BaseLayer>();
            BatchSize = batchSize;
            _currentBatch = 0;

            CostFunction  = cost;
        }

        public void CreateLayer(ILayer layer)
        {
            if (IsLocked) throw new InvalidOperationException("Network is Locked.");

            if (((BaseLayer)layer).OptimizerFunction.Type() == EOptimizerType.Default) ((BaseLayer)layer).OptimizerFunction = OptimizerFunction;

            Layers.Add(((BaseLayer)layer));
        }

        public void InitNetwork()
        {
            if (IsLocked) throw new InvalidOperationException("Network is Locked.");
            IsLocked = true;

            var result = new Result(Layers[^1].NeuronCount);
            Layers.Add(result);

            // Initialize All Layers, Their Ws and Bs
            for (var i = 0; i < Layers.Count - 1; i++)
            {
                // Weights
                Layers[i].Params["W"] = new Matrix(Layers[i + 1].NeuronCount, Layers[i].NeuronCount);
                
                if (i == 0 && ((BaseInitializer)Layers[i].InitializerFunction).Type() == EInitializerType.Auto)
                {
                    ((BaseInitializer)Layers[i].InitializerFunction).Scale *= Math.Sqrt(2.0 / Layers[i].NeuronCount);
                    Layers[i].Params["W"] = Layers[i].InitializerFunction.Initialize(Layers[i].Params["W"]);
                }
                else if (i != 0 && Layers[i].InitializerFunction.Type() == EInitializerType.Auto)
                {
                    ((BaseInitializer)Layers[i].InitializerFunction).Scale *= Math.Sqrt(2.0 / (Layers[i - 1].NeuronCount * Layers[i].NeuronCount));
                    Layers[i].Params["W"] = Layers[i].InitializerFunction.Initialize(Layers[i].Params["W"]);
                }
                else
                {
                    Layers[i].Params["W"] = Layers[i].InitializerFunction.Initialize(Layers[i].Params["W"]);
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

        private void ResetBatchError()
        {
            ((BaseCost)CostFunction).BatchCost = 0;
        }

        private void Backward(Matrix expected)
        {
            Y = expected;

            var da = CostFunction.Backward(_actual, expected);
            da *= BatchError + _regularizationSum;

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
                
                LastError = (float)CostFunction.Forward(_actual, expected);
                LastError += _regularizationSum;
                _regularizationSum = 0;

                _currentBatch++;
            }
            else if(_currentBatch == BatchSize)
            {
                // Weight Decay

                Backward(expected);
                foreach (var t in Layers) ((BaseOptimizer)t.OptimizerFunction).ApplyDecay();

                BatchError = ((BaseCost)CostFunction).BatchCost;
                ResetBatchError();
            }

        }
    }
}
