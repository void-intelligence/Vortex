using System;
using System.Collections.Generic;

using Nomad.Matrix;

using Vortex.Cost;
using Vortex.Cost.Utility;

using Vortex.Layer;
using Vortex.Layer.Utility;

using Vortex.Optimizer;
using Vortex.Optimizer.Utility;

using Vortex.Activation;
using Vortex.Activation.Utility;

using Vortex.Regularization;
using Vortex.Regularization.Utility;

namespace Vortex.Network
{
    public class Network
    {

        public List<BaseLayer> Layers { get; private set; }

        public BaseOptimizer OptimizerFunction { get; private set; }

        public BaseCost CostFunction { get; private set; }

        public bool IsLocked { get; private set; }

        public Network(CostSettings costSettings, OptimizerSettings optimizerSettings)
        {
            IsLocked = false;
            Layers = new List<BaseLayer>();

            CostFunction = (costSettings.Type()) switch
            {
                ECostType.CrossEntropyCost => new CrossEntropyCost((CrossEntropyCostSettings)costSettings),
                ECostType.ExponentionalCost => new ExponentialCost((ExponentionalCostSettings)costSettings),
                ECostType.GeneralizedKullbackLeiblerDivergence => new GeneralizedKullbackLeiblerDivergence((GeneralizedKullbackLeiblerDivergenceSettings)costSettings),
                ECostType.HellingerDistance => new HellingerDistance((HellingerDistanceSettings)costSettings),
                ECostType.ItakuraSaitoDistance => new ItakuraSaitoDistance((ItakuraSaitoDistanceSettings)costSettings),
                ECostType.KullbackLeiblerDivergence => new KullbackLeiblerDivergence((KullbackLeiblerDivergenceSettings)costSettings),
                ECostType.QuadraticCost => new QuadraticCost((QuadraticCostSettings)costSettings),
                _ => throw new ArgumentException("Cost Type Invalid."),
            };

            // Optimizer Function Setup
            OptimizerFunction = (optimizerSettings.Type()) switch
            {
                EOptimizerType.AdaDelta => new AdaDelta((AdaDeltaSettings)optimizerSettings),
                EOptimizerType.AdaGrad => new AdaGrad((AdaGradSettings)optimizerSettings),
                EOptimizerType.Adam => new Adam((AdamSettings)optimizerSettings),
                EOptimizerType.Adamax => new Adamax((AdamaxSettings)optimizerSettings),
                EOptimizerType.GradientDescent => new GradientDescent((GradientDescentSettings)optimizerSettings),
                EOptimizerType.Momentum => new Momentum((MomentumSettings)optimizerSettings),
                EOptimizerType.Nadam => new Nadam((NadamSettings)optimizerSettings),
                EOptimizerType.NesterovMomentum => new NesterovMomentum((NesterovMomentumSettings)optimizerSettings),
                EOptimizerType.RMSProp => new RMSProp((RMSPropSettings)optimizerSettings),
                _ => throw new ArgumentException("Optimizer Type Invalid."),
            };

        }

        public void CreateLayer(ELayerType layerType, int neuronCount, ActivationSettings activation, RegularizationSettings regularization)
        {
            if (IsLocked)
            {
                throw new InvalidOperationException("Network is Locked.");
            }

            // Regularization Setup
            BaseLayer layer = (layerType) switch
            {
                ELayerType.FullyConnected => new FullyConnected(new FullyConnectedSettings(neuronCount, activation, regularization), OptimizerFunction),
                ELayerType.Dropout => new Dropout(new DropoutSettings(neuronCount, activation, regularization), OptimizerFunction),
                ELayerType.Output => new Output(new OutputSettings(neuronCount, activation, regularization), OptimizerFunction),
                _ => throw new ArgumentException("Layer Type Invalid."),
            };

            Layers.Add(layer);
        }

        public void InitNetwork()
        {
            if (IsLocked)
            {
                throw new InvalidOperationException("Network is Locked.");
            }
            IsLocked = true;

            // Initialize All Layers, Their Ws and Bs
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                // Weights
                Layers[i].Params["W"] = new Matrix(Layers[i].NeuronCount, Layers[i + 1].NeuronCount);
                Layers[i].Params["W"].InRandomize();

                // Inputs
                Layers[i].Params["X"] = new Matrix(Layers[i].NeuronCount, 1);

                // Biases
                Layers[i].Params["B"] = new Matrix(Layers[i + 1].NeuronCount, 1);
                Layers[i].Params["B"].InRandomize();
            }
            // Final Layer
            Layers[^1].Params["W"] = new Matrix(Layers[^1].NeuronCount, Layers[^1].NeuronCount);
            Layers[^1].Params["W"].InRandomize();

            Layers[^1].Params["B"] = new Matrix(Layers[^1].NeuronCount, 1);
            Layers[^1].Params["B"].InRandomize();

        }

        private Matrix GlobalY;
        private Matrix GlobalYHat;
        private float last_err;
        private float regularizationSum;
        public Matrix Forward(Matrix input, Matrix Y)
        {
            regularizationSum = 0.0f;

            Matrix yHat = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                yHat = Layers[i].Forward(yHat);
                regularizationSum += Layers[i].RegularizationValue;
            }

            GlobalY = Y;
            GlobalYHat = yHat;

            last_err = (float)CostFunction.Forward(GlobalYHat, GlobalY);
            
            // Apply Regularization
            last_err += regularizationSum;

            return yHat;
        }

        public Matrix Backward()
        {
            Matrix da = CostFunction.Backward(GlobalYHat, GlobalY);
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Params["A-1"] = Layers[i - 1].Params["A"];
                da = Layers[i].Backward(da);
            }

            Layers[0].Params["A-1"] = Layers[0].Params["A"];
            da = Layers[0].Backward(da);

            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Optimize();
            }
            return da;
        }
    }
}
