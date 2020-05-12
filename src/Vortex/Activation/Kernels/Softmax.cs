// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Softmax : BaseActivation
    {
        public double SumExp { get; private set; }

        public Softmax()
        {
            SumExp = 0.0;
        }

        public override Matrix Forward(Matrix input)
        {
            var res = input.Duplicate();
            SumExp = 0.0;

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++) SumExp += Exp(input[i, j]);

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++) res[i, j] = Exp(input[i, j]) / SumExp;

            return res;
        }

        public override Matrix Backward(Matrix input)
        {
            var res = input.Duplicate();
            SumExp = 0.0;

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++) SumExp += Exp(input[i, j]);

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++)
                res[i, j] = Exp(input[i, j]) / SumExp * (1.0 - Exp(input[i, j]) / SumExp);

            return res;
        }

        protected override double Activate(double input)
        {
            return 0;
        }

        protected override double Derivative(double input)
        {
            return 0;
        }

        public override EActivationType Type()
        {
            return EActivationType.Softmax;
        }
    }
}
