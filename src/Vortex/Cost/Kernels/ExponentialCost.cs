// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class ExponentialCost : BaseCost
    {
        public double Tao { get; set; }

        public ExponentialCost(double tao)
        {
            Tao = tao;
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;
            
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(actual[i, j] - expected[i, j], 2);

            error /= Tao;
            error = Math.Exp(error);
            error *= Tao;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(actual[i, j] - expected[i, j], 2);

            error /= Tao;
            error = Math.Exp(error);
            error *= Tao;
            
            var gradMatrix = actual.Duplicate();
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (actual[i, j] - expected[i, j]) * error;

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.ExponentionalCost;
        }
    }
}
