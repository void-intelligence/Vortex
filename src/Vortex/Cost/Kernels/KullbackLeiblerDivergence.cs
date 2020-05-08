// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Kullback Leibler Divergence" Also known as "Information Divergence", "Information Gain", "Relative entropy", "KLIC", or "KL Divergence".
    /// </summary>
    public class KullbackLeiblerDivergenceKernel : BaseCostKernel
    {
        public KullbackLeiblerDivergenceKernel(KullbackLeiblerDivergence settings = null) : base(settings) { }

        public override double Forward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]);
                }
            }

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = -(expected[i, j] / actual[i, j]);
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.KullbackLeiblerDivergence;
    }

    public class KullbackLeiblerDivergence : BaseCost 
    {
        public override ECostType Type() => ECostType.KullbackLeiblerDivergence;
    }
}
