// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class SoftmaxKernel : BaseActivationKernel
    {
        public SoftmaxKernel(Softmax settings = null) : base(settings) { }

        public double SumExp { get; }

        public SoftmaxKernel() : base(null)
        {
            SumExp = 0.0;
        }

        public override Matrix Forward(Matrix input)
        {
            Matrix res = input.Duplicate();
            SumExp = 0.0;

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    SumExp += Exp(input[i, j]);
                }
            }

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    res[i, j] = Exp(input[i, j]) / SumExp;
                }
            }


            return res;
        }

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => 0;

        protected override double Derivative(double input) => Exp(input) / SumExp * (1 - Exp(input) / SumExp);

        public override EActivationType Type() => EActivationType.Softmax;
    }

    public sealed class Softmax : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Softmax;
    }
}
