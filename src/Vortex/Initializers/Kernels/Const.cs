using System;
using Nomad.Matrix;
using Vortex.Initializers.Utility;

namespace Vortex.Initializers.Kernels
{
    public sealed class ConstKernel : BaseInitializerKernel
    {
        public double Value;

        public ConstKernel(Const initializer) : base(initializer)
        {
            Value = initializer.Value;
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InFill(Value);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.Const;
    }

    public sealed class Const : BaseInitializer
    {
        public double Value;

        public Const(double value, double scale = 1.0) : base(scale)
        {
            Value = value;
        }

        public override EInitializerType Type() => EInitializerType.Const;
    }
}