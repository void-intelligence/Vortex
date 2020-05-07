// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Cross Entropy Cost": Also known as "Bernoulli negative log-likelihood" and "Binary Cross-Entropy"
    /// </summary>
    public class CrossEntropyCostKernel : BaseCost
    {
        public CrossEntropyCostKernel(CrossEntropyCost settings = null) : base(settings) { }

        public override double Forward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += -expected[i, j] * Math.Log(actual[i, j]) + (1.0 - expected[i, j]) * Math.Log(1 - actual[i, j]);
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
                    gradMatrix[i, j] = ((actual[i, j] - expected[i, j])) / ((1 - actual[i, j]) * actual[i, j]);
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.CrossEntropyCost;
    }

    public class CrossEntropyCost : CostSettings 
    {
        public override ECostType Type() => ECostType.CrossEntropyCost;
    }
}
