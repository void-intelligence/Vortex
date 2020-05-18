// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;

namespace Vortex.Metric.Kernels.Categorical
{
    public sealed class ArgMaxAccuracy : BaseMetric
    {
        public ArgMaxAccuracy(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            var max = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                if (actual[i, j] > max)
                    max = actual[i, j];


            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                if (Math.Abs(actual[i, j] - max) < Threshold)
                    actual[i, j] = 1.0;


            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                val += Math.Abs(actual[i, j] - expected[i, j]) < Threshold ? 0 : 1;

            return val;
        }

        public override EMetricType Type()
        {
            return EMetricType.CategoricalArgMaxAccuracy;
        }
    }
}