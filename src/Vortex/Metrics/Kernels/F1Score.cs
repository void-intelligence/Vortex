// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;

namespace Vortex.Metrics.Kernels
{
    public sealed class F1Score : BaseMetrics
    {
        public F1Score(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (Math.Abs(actual[i, j]) < Threshold && Math.Abs(expected[i, j]) < Threshold)
                {
                    val += 0.0;
                }
                else if (Math.Abs(actual[i, j] - 1.0) < Threshold && Math.Abs(expected[i, j] - 1.0) < Threshold)
                {
                    val += 2.0;
                    div += 2.0;
                }

                div++;
            }

            val /= div;
            return val;
        }

        public override EMetricsType Type()
        {
            return EMetricsType.F1Score;
        }
    }
}