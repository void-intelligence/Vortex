// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public sealed class TruncNormal : BaseInitializer
    {
        public double Mean { get; set; }
        public double Margin { get; set; }
        public double Sd { get; set; }

        public TruncNormal(double mean = 0.0, double margin = 0.1, double sd = 0.05, double scale = 1.0, double max = 1.0) : base(scale, 0, max)
        {
            Mean = mean;
            Margin = margin;
            Sd = sd;
        }

        public override Matrix Initialize(Matrix w)
        {
            var rng = new Random();
            var mat = w.Duplicate();

            for (var i = 0; i < w.Rows; i++)
            for (var j = 0; j < w.Columns; j++)
            {
                var rndVal = rng.NextDouble();
                var minVal = Exp(-0.5 * Pow((Mean - Margin) / (Sd * Sin(rndVal * 2.0 * PI)), 2.0));
                var tmp = Sd * Sqrt(-2.0 * Log(rng.NextDouble() * (Max - minVal) + minVal)) * Sin(2.0 * PI * rndVal) + Mean;
                mat[i, j] = tmp;
            }

            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Normal;
        }
    }
}