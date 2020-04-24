// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Kullback Leibler Divergence" Also known as "Information Divergence", "Information Gain", "Relative entropy", "KLIC", or "KL Divergence".
    /// </summary>
    public class KullbackLeiblerDivergence : Utility.BaseCost
    {
        public KullbackLeiblerDivergence(KullbackLeiblerDivergenceSettings settings) : base(settings) { }

        public override double Forward(Matrix Actual, Matrix Expected)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += Expected[i, j] * Math.Log(Expected[i, j] / Actual[i, j]);
                }
            }

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix Actual, Matrix Expected)
        {
            Matrix gradMatrix = Actual.Duplicate();

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    gradMatrix[i, j] = -(Expected[i, j] / Actual[i, j]);
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.KullbackLeiblerDivergence;
    }

    public class KullbackLeiblerDivergenceSettings : CostSettings 
    {
        public override ECostType Type() => ECostType.KullbackLeiblerDivergence;
    }
}
