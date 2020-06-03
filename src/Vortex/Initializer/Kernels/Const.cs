// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public sealed class Const : BaseInitializer
    {
        public double Value { get; set; }

        public Const(double value = 1.0, double scale = 1.0) : base(scale, 0, 1)
        {
            Value = value;
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InFill(Value);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Const;
        }
    }
}