// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Activation.Kernels;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;
using System.Collections.Generic;
using Vortex.Optimizer.Utility;
using Vortex.Regularization.Kernels;
using Vortex.Initializer.Kernels;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Mutation.Kernels;

namespace Vortex.Layer.Utility
{
    /// <summary>
    /// Base of all Layer classes
    /// </summary>
    public abstract class BaseLayerKernel
    {
        public float RegularizationValue { get; protected set; }
        public BaseLayer Settings { get; }
        public int NeuronCount { get; }
        public BaseActivationKernel ActivationFunction { get; }
        public BaseRegularizationKernel RegularizationFunction { get; }
        public BaseOptimizerKernel OptimizerFunction { get; set; }
        public BaseInitializerKernel Initializer { get; }
        public BaseMutationKernel MutationFunction { get; }
        public Dictionary<string, Matrix> Params { get; }
        public Dictionary<string, Matrix> Grads { get; }

        protected BaseLayerKernel(BaseLayer layerSettings)
        {
            RegularizationValue = 0.0f;

            Params = new Dictionary<string, Matrix>();

            Grads = new Dictionary<string, Matrix>();

            Settings = layerSettings;

            NeuronCount = Settings.NeuronCount;

            MutationFunction = (Settings.MutationFunction.Type()) switch
            {
                EMutationType.DefaultMutation => new DefaultMutationKernel(),
                EMutationType.NoMutation => new NoMutationKernel(),
                _ => throw new ArgumentException("Mutation Type Invalid.")

            };

            // Initializer Setup
            Initializer = (Settings.InitializerFunction.Type()) switch
            {
                EInitializerType.Const => new ConstKernel((Const)Settings.InitializerFunction),
                EInitializerType.GlorotNormal => new GlorotNormalKernel((GlorotNormal)Settings.InitializerFunction),
                EInitializerType.GlorotUniform => new GlorotUniformKernel((GlorotUniform)Settings.InitializerFunction),
                EInitializerType.HeNormal => new HeNormalKernel((HeNormal)Settings.InitializerFunction),
                EInitializerType.HeUniform => new HeUniformKernel((HeUniform)Settings.InitializerFunction),
                EInitializerType.LeCunNormal => new LeCunNormalKernel((LeCunNormal)Settings.InitializerFunction),
                EInitializerType.LeCunUniform => new LeCunUniformKernel((LeCunUniform)Settings.InitializerFunction),
                EInitializerType.Normal => new NormalKernel((Normal)Settings.InitializerFunction),
                EInitializerType.One => new OneKernel((One)Settings.InitializerFunction),
                EInitializerType.Uniform => new UniformKernel((Uniform)Settings.InitializerFunction),
                EInitializerType.Zero => new ZeroKernel((Zero)Settings.InitializerFunction),
                _ => throw new ArgumentException("Initializer Type Invalid.")
            };


            // Activation Setup
            ActivationFunction = (Settings.ActivationFunction.Type()) switch
            {
                EActivationType.Arctan => new ArctanKernel(),
                EActivationType.BinaryStep => new BinaryStepKernel(),
                EActivationType.BipolarSigmoid => new BipolarSigmoidKernel(),
                EActivationType.Elu => new EluKernel((Elu)Settings.ActivationFunction),
                EActivationType.Exponential => new ExponentialKernel(),
                EActivationType.HardSigmoid=> new HardSigmoidKernel(),
                EActivationType.HardTanh=> new HardTanhKernel(),
                EActivationType.Identity=> new IdentityKernel(),
                EActivationType.Loggy => new LoggyKernel(),
                EActivationType.Logit => new LogitKernel(),
                EActivationType.LRelu => new LReluKernel((LRelu)Settings.ActivationFunction),
                EActivationType.Mish => new MishKernel(),
                EActivationType.Relu => new ReluKernel(),
                EActivationType.Selu => new SeluKernel(),
                EActivationType.Sigmoid => new SigmoidKernel(),
                EActivationType.Softmax => new SoftmaxKernel(),
                EActivationType.Softplus => new SoftplusKernel(),
                EActivationType.Softsign => new SoftsignKernel(),
                EActivationType.Swish => new SwishKernel(),
                EActivationType.Tanh => new TanhKernel(),
                _ => throw new ArgumentException("Activation Type Invalid."),
            };

            // Regularization Setup
            RegularizationFunction = (Settings.RegularizationFunction.Type()) switch
            {
                ERegularizationType.None => new NoneKernel((None)Settings.RegularizationFunction),
                ERegularizationType.L1 => new L1Kernel((L1)Settings.RegularizationFunction),
                ERegularizationType.L2 => new L2Kernel((L2)Settings.RegularizationFunction),
                _ => throw new ArgumentException("Regularization Type Invalid."),
            };
        }

        // All Layer Forward Calculations
        public abstract Matrix Forward(Matrix inputs);

        // All Layer Backward Calculations
        public abstract Matrix Backward(Matrix dA);

        // All Layer Optimization Calculations
        public abstract void Optimize();

        public abstract ELayerType Type();
    }

    public abstract class BaseLayer
    {
        public int NeuronCount { get; }
        public BaseActivation ActivationFunction { get; }
        public BaseRegularization RegularizationFunction { get; }
        public BaseInitializer InitializerFunction { get; }
        public BaseMutation MutationFunction { get; }

#nullable enable
        protected BaseLayer(int neuronCount, BaseActivation activation, BaseRegularization? regularization, BaseInitializer? initializer, BaseMutation? mutation) 
        {
            NeuronCount = neuronCount;
            ActivationFunction = activation;
            RegularizationFunction = regularization ?? new None();
            InitializerFunction = initializer ?? new Auto();
            MutationFunction = mutation ?? new NoMutation();
        }
#nullable disable

        public abstract ELayerType Type();
    }
}
