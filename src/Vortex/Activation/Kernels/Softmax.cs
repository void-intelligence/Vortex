// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Softmax : BaseActivation
    {
        public double SumExp { get; private set; }

        private Matrix CachedSoftmax;

        public Softmax()
        {
            SumExp = 0.0;
        }

        public override Matrix Forward(Matrix input)
        {
            SumExp = 0.0;
            SumExp = input.Map(Exp).Sum();
            CachedSoftmax = input.Map(Exp) / SumExp;
            return CachedSoftmax;
        }

        public override Matrix Backward(Matrix input)
        {
            // Softmax Derivative
            Forward(input);

            // Jacobian Matrix
            input.InFlatten();
            var jac = new Matrix(input.Rows, input.Rows).Fill(0);
            for (var i = 0; i < CachedSoftmax.Rows; i++) jac[i, i] = input[i, 0];
            for (var i = 0; i < jac.Rows; i++)
            for (var j = 0; j < jac.Columns; j++)
                if (i == j) jac[i, j] = CachedSoftmax[i, 0] * (1 - CachedSoftmax[j, 0]);
                else jac[i, j] = -CachedSoftmax[i, 0] * CachedSoftmax[j, 0];
            return jac;
        }

        public override double Activate(double input)
        {
            throw new InvalidOperationException("Cannot operate softmax on a single value!");
        }

        public override double Derivative(double input)
        {
            throw new InvalidOperationException("Cannot operate softmax derivative on a single value!");
        }

        public override EActivationType Type()
        {
            return EActivationType.Softmax;
        }
    }
}
