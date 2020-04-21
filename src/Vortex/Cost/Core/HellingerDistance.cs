// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Hellinger Distance": needs to have positive values, and ideally values between 0 and 1. The same is true for the following divergences.
    /// </summary>
    public class HellingerDistance : Utility.BaseCost
    {
        public HellingerDistance(HellingerDistanceSettings settings) : base(settings) { }
        
        public override double Forward(Matrix Actual, Matrix Expected, int layerCount)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += Math.Pow((Math.Sqrt(Actual[i, j]) - Math.Sqrt(Expected[i, j])), 2);
                }
            }

            error *= (1 / Math.Sqrt(2));
            
            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix Actual, Matrix Expected, int layerCount)
        {
            if (Actual.Rows != Expected.Rows || Actual.Columns != Expected.Columns)
            {
                throw new ArgumentException("Actual Matrix does not have the same size as The Expected Matrix");
            }

            Matrix gradMatrix = Actual.Duplicate();

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    gradMatrix[i, j] = (Math.Sqrt(Actual[i, j]) - Math.Sqrt(Expected[i, j])) / (Math.Sqrt(2) * Math.Sqrt(Actual[i, j]));
                }
            }

            return gradMatrix;
        }

        public override string ToString()
        {
            return Type().ToString();
        }

        public override ECostType Type()
        {
            return ECostType.HellingerDistance;
        }
    }

    public class HellingerDistanceSettings : CostSettings { }
}
