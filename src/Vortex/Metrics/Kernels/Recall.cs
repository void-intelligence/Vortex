// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;

namespace Vortex.Metrics.Kernels
{
    public sealed class Recall : BaseMetrics
    {
        public Recall(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (!(Math.Abs(expected[i, j] - 1.0) < 0.1)) continue;
                div++;
                if (actual[i, j] >= Threshold)
                    val++;
            }

            val /= div;
            return val;
        }

        public override EMetricsType Type()
        {
            return EMetricsType.Recall;
        }
    }
}