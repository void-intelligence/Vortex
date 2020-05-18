// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;

namespace Vortex.Metric.Kernels.Categorical
{
    public sealed class Accuracy : BaseMetric
    {
        public Accuracy(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                val += Math.Abs(actual[i, j] - expected[i, j]) < Threshold ? 0 : 1;

            val /= (actual.Rows * actual.Columns);
            return val;
        }

        public override EMetricType Type()
        {
            return EMetricType.CategoricalAccuracy;
        }
    }
}