// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class ExponentialCostKernel : BaseCostKernel
    {
        public double Tao { get; set; }

        public ExponentialCostKernel(ExponentionalCost settings) : base(settings)
        {
            Tao = settings.Tao;
        }

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

            error /= Tao;

            error = Math.Exp(error);

            error *= Tao;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= Tao;

            error = Math.Exp(error);

            error *= Tao;
            
            Matrix gradMatrix = actual.Duplicate();
            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (actual[i, j] - expected[i, j]) * error;
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.ExponentionalCost;
    }

    public class ExponentionalCost : BaseCost
    {
        public double Tao { get; set; }

        public ExponentionalCost(double tao) { Tao = tao; }

        public override ECostType Type() => ECostType.ExponentionalCost;
    }
}
