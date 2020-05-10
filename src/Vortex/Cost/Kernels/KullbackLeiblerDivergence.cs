// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Kullback Leibler Divergence" Also known as "Information Divergence", "Information Gain", "Relative entropy", "KLIC", or "KL Divergence".
    /// </summary>
    public sealed class KullbackLeiblerDivergence : BaseCost
    {
        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]);

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = -(expected[i, j] / actual[i, j]);

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.KullbackLeiblerDivergence;
        }
    }
}
