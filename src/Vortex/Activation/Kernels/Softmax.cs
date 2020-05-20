// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
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
            // Jacobian Matrix
            input.InFlatten();
            var jac = new Matrix(input.Rows, input.Rows).Fill(0);
            
            for (var i = 0; i < input.Rows; i++) jac[i, i] = input[i,0];
            for (var i = 0; i < jac.Rows; i++)
            for (var j = 0; j < jac.Columns; j++)
                if (i == j) jac[i, j] = input[i, 0] * (1 - input[j, 0]);
                else jac[i, j] = -input[i, 0] * input[j, 0];
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
