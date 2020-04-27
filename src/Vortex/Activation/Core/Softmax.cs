// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Softmax : Utility.BaseActivation
    {
        public Softmax(SoftmaxSettings settings = null) : base(settings) { }

        public double SumExp { get; private set; }

        public Softmax() : base(null)
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
                    SumExp += System.Math.Exp(input[i, j]);
                }
            }

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    res[i, j] = System.Math.Exp(input[i, j]) / SumExp;
                }
            }


            return res;
        }

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => 0;

        protected override double Derivative(double input) => System.Math.Exp(input) / SumExp * (1 - System.Math.Exp(input) / SumExp);

        public override Utility.EActivationType Type() => Utility.EActivationType.Softmax;

        public override string ToString() => Type().ToString();
    }

    public sealed class SoftmaxSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Softmax;
    }
}
