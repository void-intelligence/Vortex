// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;

using Nomad.Matrix;
using Vortex.Cost.Utility;

using Vortex.Layer.Kernels;
using Vortex.Layer.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;
using Vortex.Metrics.Kernels;
using Vortex.Metrics.Utility;
using Vortex.Optimizer.Kernels;

namespace Vortex.Network
{
    public class Sequential
    {
        // Helpers
        public double BatchError { get; private set; }
        public double LastError { get; private set; }
        public Matrix Y { get; private set; }
        public bool IsLocked { get; private set; }
        public int BatchSize { get; set; }

        // Component Properties
        public List<BaseLayer> Layers { get; }
        public IMetrics MetricsFunction { get; private set; }
        public IOptimizer OptimizerFunction { get; }
        public ICost CostFunction { get; }

        // Train Utility
        public Matrix LastOutput { get; private set; }
        public float LastRegularizationSum { get; private set; }


#nullable enable
        public Sequential(ICost cost, IOptimizer optimizer, IMetrics? metrics = null, int batchSize = 1)
        {
            // If Optimizer is set to default use Adam
            OptimizerFunction = optimizer.Type() == EOptimizerType.Default ? new Adam(((BaseOptimizer)optimizer).Alpha, ((BaseOptimizer)optimizer).Decay) : optimizer;
            MetricsFunction = metrics ?? new Accuracy();

            IsLocked = false;
            Layers = new List<BaseLayer>();
            BatchSize = batchSize;
            _currentBatch = 0;

            CostFunction  = cost;
        }
#nullable disable

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


        public Matrix Forward(Matrix input)
        {
            LastRegularizationSum = 0.0f;

            var yHat = input;
            foreach (var t in Layers)
            {
                yHat = t.Forward(yHat);
                LastRegularizationSum += t.RegularizationValue;
            }

            // Save yHat (last output)
            LastOutput = yHat;
            return yHat;
        }

#nullable enable
        public double Train(Matrix input, Matrix expected, IMetrics? metrics = null)
        {
            if (metrics != null)
            {
                MetricsFunction = metrics;
            }

            var yHat = input;
            if (_currentBatch < BatchSize)
            {
                #region Feed Forward
                // Forward
                LastRegularizationSum = 0.0f;
                foreach (var t in Layers)
                {
                    yHat = t.Forward(yHat);
                    LastRegularizationSum += t.RegularizationValue;
                }
                LastOutput = yHat;

                // Error Calculation
                LastError = (float)CostFunction.Forward(LastOutput, expected);
                LastError += LastRegularizationSum;
                LastRegularizationSum = 0;
                _currentBatch++;
                #endregion
            }
            else if(_currentBatch == BatchSize)
            {
                // Save Y
                Y = expected;

                #region Backprop

                // Calculate Final Layer dA (derivative of Cost * J)
                var da = CostFunction.Backward(LastOutput, expected);
                da *= BatchError + LastRegularizationSum;

                // Calculate da dw db for all layers
                for (var i = Layers.Count - 2; i >= 0; i--) da = Layers[i].Backward(da);

                #endregion

                #region Optimization

                // Optimize all layers
                foreach (var t in Layers) t.Optimize();

                // Weight Decay
                foreach (var t in Layers) ((BaseOptimizer)t.OptimizerFunction).ApplyDecay();

                #endregion
                
                // Reset Error for the next Batch
                BatchError = ((BaseCost)CostFunction).BatchCost;
                ((BaseCost)CostFunction).BatchCost = 0;
            }

            // Metrics
            return MetricsFunction.Evaluate(LastOutput, expected);
        }
#nullable disable
    }
}
