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
            SumExp = 0.0;
            SumExp = input.Map(Exp).Sum();
            return input.Map(Exp) / SumExp;
        }

        public override Matrix Backward(Matrix input)
        {
            // Softmax Derivative

            // Diagflat
            input.InFlatten();
            var diagFlat = new Matrix(input.Rows, input.Rows).Fill(0);
            for (var i = 0; i < input.Rows; i++) diagFlat[i, i] = input[i, 0];
            for (var i = 0; i < diagFlat.Rows; i++)
            for (var j = 0; j < diagFlat.Columns; j++)
                if (i == j)
                    diagFlat[i, j] = input[i, 0] * (1 - input[j, 0]);
            //else diagFlat[i, j] = -input[i, 0] * input[j, 0];

            return diagFlat - input * input.T();
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
