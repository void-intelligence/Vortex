// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Quadratic Cost": Also known as "Mean Squared Error" or "Maximum Likelihood" or "Sum Squared Error"
    /// </summary>
    public class QuadraticCostKernel : BaseCost
    {
        public QuadraticCostKernel(QuadraticCost settings = null) : base(settings) { }
        
        public override double Forward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= 2;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            int n = 0;
            Matrix da = actual.Duplicate();
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                n++;
                da[i, j] = Math.Pow(expected[i, j] - actual[i, j], 2);
            }

            return da * (1.0 / n);
        }

        public override ECostType Type() => ECostType.QuadraticCost;
    }

    public class QuadraticCost : CostSettings
    {
        public override ECostType Type() => ECostType.QuadraticCost;
    }
}
